from .base_options import BaseOptions
from .base_options import BaseOptionsObj


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='main', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [self.checkpoints_dir]/[self.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--lr_decay', type=float, default=1, help='learning rate decay for certain module ...')

        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        self.isTrain = True
        return parser


class TrainOptionsObj(BaseOptionsObj):
    display_freq = 400
    display_ncols = 4
    display_env = 'main'
    display_id = 1
    display_server = "http://localhost"
    display_port = 8097
    update_html_freq = 1000
    print_freq = 100
    no_html = True
    save_latest_freq = 200
    save_epoch_freq = 5
    save_by_iter = False
    epoch_count = 1
    lr_decay = 1
    phase = 'train'
    niter = 100
    niter_decay = 100
    beta1 = 0.5
    lr = 0.0002
    lr_policy = 'linear'
    lr_decay_iters = 50
    isTrain = True

    def __init__(self,
                 name=None,
                 data=None,
                 epochs=None,
                 project=None,
                 quick=False,
                 weights=None,
                 output=None,
                 statusbar=None,
                 batch_size=4,
                 lr=0.001,
                 chip_size=256,
                 angle=15,
                 model_name='CDFA',
                 SA_mode='PAM',
                 preprocess='rotate_and_crop',
                 save_epoch_freq=5,
                 resume=False):
        super(TrainOptionsObj, self).__init__()
        self.name = name
        self.epochs = epochs
        self.dataroot = data
        self.val_dataroot = data
        self.batch_size = batch_size
        self.lr = lr
        self.crop_size = chip_size
        self.anlge = angle
        self.model = model_name
        self.SA_mode = SA_mode
        self.preprocess = preprocess
        self.save_epoch_freq = save_epoch_freq
        self.statusbar = statusbar
        self.continue_train = resume
        self.checkpoints_dir = project
        self.output = output
        self.weights = weights
