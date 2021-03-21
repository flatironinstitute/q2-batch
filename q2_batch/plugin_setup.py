import importlib
import qiime2.plugin
import qiime2.sdk
from qiime2.plugin import (Str, Properties, Int, Float,  Metadata, Bool,
                           MetadataColumn, Categorical)

from q2_batch import __version__
from q2_batch._method import estimate, slurm_estimate
from q2_differential._type import FeatureTensor
from q2_differential._format import FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt
from q2_types.feature_table import FeatureTable, Frequency


plugin = qiime2.plugin.Plugin(
    name='batch',
    version=__version__,
    website="https://github.com/mortonjt/q2-batch",
    citations=[],
    short_description=('Plugin for quick and dirty batch effect estimation'),
    description=('This is a QIIME 2 plugin for estimating batch effects'
                 ' for downstream plugins'),
    package='q2-batch')

plugin.methods.register_function(
    function=estimate,
    inputs={'counts': FeatureTable[Frequency]},
    parameters={
        'batches': MetadataColumn[Categorical],
        'replicates': MetadataColumn[Categorical],
        'monte_carlo_samples': Int,
        'cores': Int
    },
    outputs=[
        ('posterior', FeatureTensor)
    ],
    input_descriptions={
        "counts": "Input table of counts.",
    },
    output_descriptions={
        'posterior': ('Output posterior distribution of batch effect'),
    },
    parameter_descriptions={
        'batches': ('Specifies the batch ids'),
        'replicates': ('Specifies the technical replicates.'),
        'monte_carlo_samples': (
            'Number of monte carlo samples to draw from '
            'posterior distribution.'
        ),
        'cores' : 'Number of cpu cores'
    },
    name='estimation',
    description=("Computes batch effects from technical replicates"),
    citations=[]
)


plugin.methods.register_function(
    function=slurm_estimate,
    inputs={'counts': FeatureTable[Frequency]},
    parameters={
        'batches': MetadataColumn[Categorical],
        'replicates': MetadataColumn[Categorical],
        'monte_carlo_samples': Int,
        'cores': Int,
        'processes': Int,
        'nodes': Int,
        'memory': Str,
        'walltime': Str,
        'queue': Str
    },
    outputs=[
        ('posterior', FeatureTensor)
    ],
    input_descriptions={
        "counts": "Input table of counts.",
    },
    output_descriptions={
        'posterior': ('Output posterior distribution of batch effect'),
    },
    parameter_descriptions={
        'batches': ('Specifies the batch ids'),
        'replicates': ('Specifies the technical replicates.'),
        'monte_carlo_samples': (
            'Number of monte carlo samples to draw from '
            'posterior distribution.'
        ),
        'cores' : 'Number of cpu cores per process',
        'processes' : 'Number of processes',
        'nodes' : 'Number of nodes',
        'memory' : "Amount of memory per process (default: '16GB'",
        'walltime' : "Amount of time to spend on each worker (default : '01:00:00')",
        'queue' : "Processing queue"
    },
    name='parallel estimation',
    description=("Computes batch effects from technical replicates on a slurm cluster"),
    citations=[]
)
