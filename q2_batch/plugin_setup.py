import importlib
import qiime2.plugin
import qiime2.sdk
from qiime2.plugin import (Str, Properties, Int, Float,  Metadata, Bool,
                           MetadataColumn, Categorical, Continuous)

from q2_fido import __version__
from q2_differential._type import FeatureTensor
from q2_differential._format import FeatureTensorNetCDFFormat, FeatureTensorNetCDFDirFmt
from q2_fido._method import basset


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
    function=dirichlet_multinomial,
    inputs={'table': FeatureTable[Frequency]},
    parameters={
        'batches': MetadataColumn[Categorical],
        'replicates': MetadataColumn[Categorical],
        'monte_carlo_samples': Int
    },
    outputs=[
        ('posterior', FeatureTensor)
    ],
    input_descriptions={
        "table": "Input table of counts.",
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

    },
    name='estimation',
    description=("Computes batch effects from technical replicates"),
    citations=[]
)
