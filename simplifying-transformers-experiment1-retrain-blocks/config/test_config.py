from config.base_config import BaseConfig


class TestConfig(BaseConfig):

    def __init__(self):
        super(BaseConfig).__init__()
        self.name = 'test options'
        parser = BaseConfig().get_parser()

        parser.add_argument('--train', dest='train', action='store_true')
        parser.add_argument('--percentage_data', type=float, default=0.5,
                            help='% Amount of Data used to evaluate the Model')
        parser.set_defaults(train=False)

        self.parser = parser

    def get_parser(self):
        return self.parser

    def get(self):
        return self.parser.parse_args()
