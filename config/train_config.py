from config.base_config import BaseConfig
from config.block_config import BlockConfig


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


class BlockTrainConfig(BlockConfig):

    def __init__(self):
        super(BlockConfig).__init__()
        self.name = 'training options'
        parser = BlockConfig().get_parser()

        parser.add_argument('--train', dest='train', action='store_true')
        parser.set_defaults(train=True)
        parser.add_argument('--model_checkpoint', type=str, default="models/_checkpoints/wikitext2/BERTLM-latest.pth",
                            help="this checkpoint is loaded to the teacher before training")
        parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train for.')

        parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
        parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
        parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps before weight decay.")

        parser.add_argument("--block", type=int, default=12, help="Layer in which the block to train is")

        parser.add_argument('--save_checkpoint_every', type=int, default=5, help='specify how often a lasting'
                                                                                 'checkpoint should be stored.')

        self.parser = parser

    def get_parser(self):
        return self.parser

    def get(self):
        return self.parser.parse_args()
