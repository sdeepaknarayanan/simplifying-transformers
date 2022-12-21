from config.base_config import BaseConfig


class TrainConfig(BaseConfig):

    def __init__(self):
        super(BaseConfig).__init__()
        self.name = 'training options'
        parser = BaseConfig().get_parser()

        parser.add_argument('--train', dest='train', action='store_true')
        parser.set_defaults(train=True)

        parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train for.')

        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
        parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
        parser.add_argument("--warmup_steps", type=int, default=1000, help="warmup steps before weight decay.")

        parser.add_argument('--save_checkpoint_every', type=int, default=5, help='specify how often a lasting'
                                                                                 'checkpoint should be stored.')

        self.parser = parser

    def get_parser(self):
        return self.parser

    def get(self):
        return self.parser.parse_args()