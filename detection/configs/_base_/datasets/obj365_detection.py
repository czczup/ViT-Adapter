# dataset settings
dataset_type = 'CocoDataset'
classes = ['Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp',
           'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf',
           'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet',
           'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower',
           'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots',
           'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass',
           'Belt', 'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light',
           'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle',
           'Stool', 'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum',
           'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle', 'Guitar',
           'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck',
           'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle',
           'Sailboat', 'Laptop', 'Awning', 'Bed', 'Faucet', 'Tent', 'Horse',
           'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife',
           'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign',
           'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake',
           'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin',
           'Other Fish', 'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato',
           'Lantern', 'Machinery Vehicle', 'Fan', 'Green Vegetables', 'Banana',
           'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer',
           'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley',
           'Head Phone', 'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller',
           'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat',
           'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant',
           'Skateboard', 'Surfboard', 'Gun', 'Skating and Skiing shoes', 'Gas stove',
           'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite', 'Strawberry', 'Other Balls',
           'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products',
           'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board',
           'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder',
           'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra',
           'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg',
           'Fire Extinguisher', 'Candy', 'Fire Truck', 'Billards', 'Converter',
           'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette ',
           'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord',
           'Tong', 'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask',
           'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage',
           'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine/Drying Machine',
           'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush',
           'Ice cream', 'Hotair ballon', 'Cello', 'French Fries', 'Scale', 'Trophy',
           'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball',
           'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball',
           'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal',
           'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan',
           'Helicopter', 'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker',
           'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit', 'Router/modem',
           'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry',
           'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask',
           'Mushroon', 'Screwdriver', 'Soap', 'Recorder', 'Bear', 'Eggplant', 'Board Eraser',
           'Coconut', 'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak',
           'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate', 'Dishwasher',
           'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya',
           'Antelope', 'Parrot', 'Seal', 'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal',
           'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill',
           'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French',
           'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars',
           'Asparagus', 'Barbell', 'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster',
           'Table Teniis paddle', 'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser',
           'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis ']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
file_client_args = dict(backend='petrel')
train_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=file_client_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='AutoAugment',
        policies=[
            [
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    keep_ratio=True)
            ],
            [
                dict(
                    type='Resize',
                    # The radio of all image in train dataset < 7
                    # follow the original impl
                    img_scale=[(400, 4200), (500, 4200), (600, 4200)],
                    multiscale_mode='value',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 600),
                    allow_negative_crop=False),
                dict(
                    type='Resize',
                    img_scale=[(480, 1333), (512, 1333), (544, 1333),
                               (576, 1333), (608, 1333), (640, 1333),
                               (672, 1333), (704, 1333), (736, 1333),
                               (768, 1333), (800, 1333)],
                    multiscale_mode='value',
                    override=True,
                    keep_ratio=True)
            ]
        ]),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='ClassBalancedDataset',
        oversample_thr=1e-3,
        dataset=dict(
            type=dataset_type,
            ann_file='/mnt/petrelfs/share_data/chenzhe1/bigdet/Objects365/zhiyuan_objv2_train.json',  # _Tiny
            img_prefix='/mnt/petrelfs/share_data/chenzhe1/bigdet/Objects365/',
            classes=classes,
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file='/mnt/petrelfs/share_data/chenzhe1/bigdet/Objects365/zhiyuan_objv2_val.json',
        img_prefix='/mnt/petrelfs/share_data/chenzhe1/bigdet/Objects365/val/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file='/mnt/petrelfs/share_data/chenzhe1/bigdet/Objects365/zhiyuan_objv2_val.json',
        img_prefix='/mnt/petrelfs/share_data/chenzhe1/bigdet/Objects365/val/',
        classes=classes,
        pipeline=test_pipeline)
)
evaluation = dict(metric='bbox')