import biom
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from skbio.stats.composition import ilr_inv
from cmdstanpy import CmdStanModel
from dask.distributed import Client, LocalCluster
from birdman import BaseModel
import tempfile
import json
import time


def _extract_replicates(replicates, batches):
    replicate_encoder = LabelEncoder()
    replicate_encoder.fit(replicates)
    replicate_ids = replicate_encoder.transform(replicates)
    # identify reference replicates - these will be the
    # first sample for each replicate group
    ref_ids, lookup = np.zeros(len(replicate_ids)), {}
    for i, c in enumerate(replicate_ids):
        if c not in lookup:
            lookup[c] = i
    for i, c in enumerate(replicate_ids):
        ref_ids[i] = lookup[c]
    batch_encoder = LabelEncoder()
    batch_encoder.fit(batches)
    batch_ids = batch_encoder.transform(batches)

    batch_ids = batch_ids.astype(np.int64) + 1
    ref_ids = ref_ids.astype(np.int64) + 1
    return ref_ids, replicate_ids, batch_ids


def _batch_func(counts: np.array, replicates: np.array,
                batches: np.array, depth: int,
                mc_samples: int = 1000) -> dict:

    ref_ids, replicate_ids, batch_ids = _extract_replicates(
        replicates, batches)
    # Actual stan modeling
    code = os.path.join(os.path.dirname(__file__),
                        'assets/batch_pln_single.stan')
    sm = CmdStanModel(stan_file=code)
    dat = {
        'N': len(counts),
        'R': int(max(replicate_ids) + 1),
        'B': int(max(batch_ids) + 1),
        'depth': list(np.log(depth)),
        'y': list(map(int, counts.astype(np.int64))),
        'ref_ids': list(map(int, ref_ids)),
        'batch_ids': list(map(int, batch_ids))
    }
    with tempfile.TemporaryDirectory() as temp_dir_name:
        data_path = os.path.join(temp_dir_name, 'data.json')
        with open(data_path, 'w') as f:
            json.dump(dat, f)
        # Obtain an initial guess with MLE
        # guess = sm.optimize(data=data_path, inits=0)
        # see https://mattocci27.github.io/assets/poilog.html
        # for recommended parameters for poisson log normal
        fit = sm.sample(data=data_path, iter_sampling=mc_samples,
                        # inits=guess.optimized_params_dict,
                        chains=4, iter_warmup=mc_samples // 2,
                        adapt_delta=0.9, max_treedepth=20)
        fit.diagnose()
        mu = fit.stan_variable('mu')
        sigma = fit.stan_variable('sigma')
        disp = fit.stan_variable('disp')
        res = pd.DataFrame({
            'mu': mu,
            'sigma': sigma,
            'disp': disp})
        # TODO: this doesn't seem to work atm, but its fixed upstream
        # res = fit.summary()
        return res


class PoissonLogNormalBatch(BaseModel):
    """Fit Batch effects estimator with Poisson Log Normal

    Parameters:
    -----------
    table: biom.table.Table
        Feature table (features x samples)
    batch_column : str
        Column that specifies `batches` of interest that
        cause technical artifacts.
    replicate_column : str
        Column that specifies technical replicates that
        are spread across batches.
    metadata: pd.DataFrame
        Metadata for matching and status covariates.
    num_iter: int
        Number of posterior sample draws, defaults to 1000
    num_warmup: int
        Number of posterior draws used for warmup, defaults to 500
    chains: int
        Number of chains to use in MCMC, defaults to 4
    seed: float
        Random seed to use for sampling, defaults to 42
    mu_scale : float
        Standard deviation for prior distribution for mu
    sigma_scale : float
        Standard deviation for prior distribution for sigma
    disp_scale : float
        Standard deviation for prior distribution for disp
    reference_scale : float
        Standard deviation for prior distribution for reference
    """
    def __init__(self,
                 table: biom.table.Table,
                 batch_column: str,
                 replicate_column: str,
                 metadata: pd.DataFrame,
                 num_iter: int = 1000,
                 num_warmup: int = 500,
                 adapt_delta: float = 0.9,
                 max_treedepth: float = 20,
                 chains: int = 4,
                 seed: float = 42,
                 mu_scale: float = 1,
                 sigma_scale: float = 1,
                 disp_scale: float = 1,
                 reference_scale: float = 10):
        model_path = os.path.join(os.path.dirname(__file__),
                                  'assets/batch_pln_single.stan')
        super(PoissonLogNormalBatch, self).__init__(
            table, metadata, model_path,
            num_iter, num_warmup, chains, seed,
            parallelize_across="features")
        # assemble replicate and batch ids
        metadata = metadata.loc[table.ids()]
        depth = table.sum(axis='sample')
        replicates = metadata[replicate_column]
        batches = metadata[batch_column]
        ref_ids, replicate_ids, batch_ids = _extract_replicates(
            replicates, batches)
        self.dat = {
            'D': table.shape[0],                 # number of features
            'N': table.shape[1],                 # number of samples
            'R': int(max(replicate_ids) + 1),
            'B': int(max(batch_ids) + 1),
            'depth': list(np.log(depth)),
            "y": table.matrix_data.todense().T.astype(int),
            'ref_ids': list(map(int, ref_ids)),
            'batch_ids': list(map(int, batch_ids))
        }
        param_dict = {
            "mu_scale": mu_scale,
            "sigma_scale": sigma_scale,
            "disp_scale": disp_scale,
            "reference_scale": reference_scale
        }
        self.add_parameters(param_dict)

        self.specify_model(
            params=["mu", "sigma", "disp", "batch", "reference"],
            dims={
                "beta": ["covariate", "feature"],
                "phi": ["feature"],
                "log_lhood": ["tbl_sample", "feature"],
                "y_predict": ["tbl_sample", "feature"]
            },
            coords={
                "feature": self.feature_names,
                "tbl_sample": self.sample_names
            },
            include_observed_data=True,
            posterior_predictive="y_predict",
            log_likelihood="log_lhood"
        )

    def fit_model(self, cluster_type: str = 'local',
                  sampler_args: dict = {},
                  dask_args: dict = {},
                  convert_to_inference: bool = False):
        if cluster_type == 'local':
            cluster = LocalCluster(**dask_args)
            cluster.scale(dask_args['n_workers'])
            client = Client(cluster)
        elif cluster_type == 'slurm':
            from dask_jobqueue import SLURMCluster
            cluster = SLURMCluster(**dask_args)
            cluster.scale(dask_args['n_workers'])
            client = Client(cluster)
            client.wait_for_workers(dask_args['n_workers'])
            time.sleep(60)
        super().fit_model(**sampler_args,
                          convert_to_inference=convert_to_inference)


def _simulate(n=100, d=10, depth=50):
    """ Simulate batch effects from Multinomial distribution

    Parameters
    ----------
    n : int
       Number of samples
    d : int
       Number of microbes
    depth : int
       Sequencing depth

    Returns
    -------
    table : pd.DataFrame
        Simulated counts
    md : pd.DataFrame
        Simulated metadata
    """
    noise = 0.1
    batch = np.random.randn(d - 1) * 10
    diff = np.random.randn(d - 1)
    ref = np.random.randn(d - 1)
    table = np.zeros((n, d))
    batch_md = np.zeros(n)
    diff_md = np.zeros(n)
    rep_md = np.zeros(n)
    for i in range(n // 2):
        delta = np.random.randn(d - 1) * noise
        N = np.random.poisson(depth)
        if i % 2 == 0:
            r1 = ref + delta
            r2 = r1 + batch
            p1 = np.random.dirichlet(ilr_inv(r1))
            p2 = np.random.dirichlet(ilr_inv(r2))
            batch_md[i] = 0
            batch_md[(n // 2) + i] = 1
            diff_md[i] = 0
            diff_md[(n // 2) + i] = 0
            rep_md[i] = i
            rep_md[(n // 2) + i] = i
            table[i] = np.random.multinomial(N, p1)
            table[(n // 2) + i] = np.random.multinomial(N, p2)
        elif i % 2 == 1:
            r1 = ref + delta + diff
            r2 = r1 + batch + diff
            p1 = np.random.dirichlet(ilr_inv(r1))
            p2 = np.random.dirichlet(ilr_inv(r2))
            batch_md[i] = 0
            batch_md[(n // 2) + i] = 1
            diff_md[i] = 1
            diff_md[(n // 2) + i] = 1
            rep_md[i] = i
            rep_md[(n // 2) + i] = i
            table[i] = np.random.multinomial(N, p1)
            table[(n // 2) + i] = np.random.multinomial(N, p2)
    oids = [f'o{x}' for x in range(d)]
    sids = [f's{x}' for x in range(n)]
    table = pd.DataFrame(table, index=sids, columns=oids)
    md = pd.DataFrame({'batch': batch_md.astype(np.int64).astype(np.str),
                       'diff': diff_md.astype(np.int64).astype(np.str),
                       'reps': rep_md.astype(np.int64).astype(np.str)},
                      index=sids)
    md.index.name = 'sampleid'
    return table, md
