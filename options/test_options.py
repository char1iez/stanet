from .base_options import BaseOptions
from .base_options import BaseOptionsObj


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        # Dropout and Batchnorm has different behavioir during training and test.
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        # To avoid cropping, the load_size should be the same as crop_size
        parser.set_defaults(load_size=parser.get_default('crop_size'))
        self.isTrain = False
        return parser


class TestOptionsObj(BaseOptionsObj):
    ntest = float('inf')
    results_dir = './results'
    aspect_ratio = 1.0
    phase = 'test'
    eval = True
    num_test = float('inf')
    load_size = 256
    dataset_mode = 'changedetection'
    isTrain = False

    def __init__(
            self,
            name=None,
            statusbar=None,
            source=None,
            output=None,
            model=None,
            model_name='CDFA',
            chip_size=480,
            format='.geojson'):
        super(TestOptionsObj, self).__init__()
        self.name = name
        self.n_class = 2
        self.SA_mode = 'PAM'
        self.arch = 'mynet3'
        self.model = model_name
        self.weights = model
        self.dataroot = source
        self.results_dir = output
        self.statusbar = statusbar
        self.chip_size = chip_size
        self.format = format
