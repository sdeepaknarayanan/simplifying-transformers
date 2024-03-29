from config.base_config import BaseConfig
from config.block_config import BlockConfig
from config.merge_config import MergeConfig


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
        parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train for.')

        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
        parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
        parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps before weight decay.")

        parser.add_argument("--block", type=int, default=10, help="Layer in which the block to train is")

        parser.add_argument('--save_checkpoint_every', type=int, default=5, help='specify how often a lasting'
                                                                                 'checkpoint should be stored.')

        self.parser = parser

    def get_parser(self):
        return self.parser

    def get(self):
        return self.parser.parse_args()


class SingleBlockRetrainConfig(BlockConfig):

    def __init__(self):
        super(BlockConfig).__init__()
        self.name = 'training options'
        parser = BlockConfig().get_parser()

        parser.add_argument('--train', dest='train', action='store_true')

        parser.add_argument('--model_checkpoint', type=str, default="models/_checkpoints/wikitext2/BERTLM-latest.pth",
                            help="this checkpoint is loaded to the teacher before training")
        parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train for.')

        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
        parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
        parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps before weight decay.")

        blocks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        parser.add_argument("--block", type=int, default=0, choices=blocks, help="Layer in which the block to train is")

        parser.add_argument('--save_checkpoint_every', type=int, default=5, help='specify how often a lasting'
                                                                                 'checkpoint should be stored.')
        criterions = ['CrossEntropy', 'MSE', 'L1', 'aRRMSE', 'CosineSimilarity']
        parser.add_argument('--criterion', type=str, default="CrossEntropy", choices=criterions, help='specify which loss to use when training')

        parser.add_argument('--loss_on_logits', dest='loss_on_scores', action='store_false')
        parser.add_argument('--loss_on_scores', dest='loss_on_scores', action='store_true')
        parser.set_defaults(train=True, print_samples=False, batch_size=64, num_workers=2, loss_on_scores=True)
        self.parser = parser

    def get_parser(self):
        return self.parser

    def get(self):
        return self.parser.parse_args()


class RetrainMergeConfig(BlockConfig):

    def __init__(self):
        super(MergeConfig).__init__()
        self.name = 'training options'
        parser = MergeConfig().get_parser()

        parser.add_argument('--train', dest='train', action='store_true')
        parser.set_defaults(train=True)
        parser.add_argument('--model_checkpoint', type=str, default="models/_checkpoints/wikitext2/BERTLM-latest.pth",
                            help="this checkpoint is loaded to the teacher before training")
        parser.add_argument('--epochs', type=int, default=15, help='Number of epochs to train for.')

        parser.add_argument("--lr", type=float, default=1e-6, help="learning rate of adam")
        parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
        parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
        parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")
        parser.add_argument("--warmup_steps", type=int, default=0, help="warmup steps before weight decay.")

        parser.add_argument("--block", type=int, default=1, help="Layer in which the block to train is")

        parser.add_argument('--save_checkpoint_every', type=int, default=10, help='specify how often a lasting'
                                                                                 'checkpoint should be stored.')

        self.parser = parser

    def get_parser(self):
        return self.parser

    def get(self):
        return self.parser.parse_args()
