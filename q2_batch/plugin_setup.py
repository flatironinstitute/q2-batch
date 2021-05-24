import importlib
import qiime2.plugin
import qiime2.sdk
from qiime2.plugin import (Str, Properties, Int, Float,  Metadata, Bool,
                           MetadataColumn, Categorical)

from q2_batch import __version__
from q2_batch._method import estimate
from q2_types.feature_data import MonteCarloTensor
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
        ('posterior', MonteCarloTensor)
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
