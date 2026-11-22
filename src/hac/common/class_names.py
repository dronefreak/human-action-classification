"""Canonical class-name lists for the datasets this project ships models for.

Every list is in Python ``sorted()`` order, which is the index order produced by
the dataset loaders (``ActionDataset`` / ``VideoDataset`` sort class directory
names). The released checkpoints were trained with that order, so index ``i`` of
a model's logits corresponds to ``LIST[i]``. Do not reorder these lists to match
other conventions (e.g. UCF-101's ``classInd.txt``), or labels will be shifted.
"""

from typing import List, Optional

STANFORD40_CLASSES = [
    "applauding",
    "blowing_bubbles",
    "brushing_teeth",
    "cleaning_the_floor",
    "climbing",
    "cooking",
    "cutting_trees",
    "cutting_vegetables",
    "drinking",
    "feeding_a_horse",
    "fishing",
    "fixing_a_bike",
    "fixing_a_car",
    "gardening",
    "holding_an_umbrella",
    "jumping",
    "looking_through_a_microscope",
    "looking_through_a_telescope",
    "phoning",
    "playing_guitar",
    "playing_violin",
    "pouring_liquid",
    "pushing_a_cart",
    "reading",
    "riding_a_bike",
    "riding_a_horse",
    "rowing_a_boat",
    "running",
    "shooting_an_arrow",
    "smoking",
    "taking_photos",
    "texting_message",
    "throwing_frisby",
    "using_a_computer",
    "walking_the_dog",
    "washing_dishes",
    "watching_TV",
    "waving_hands",
    "writing_on_a_board",
    "writing_on_a_book",
]

UCF101_CLASSES = [
    "ApplyEyeMakeup",
    "ApplyLipstick",
    "Archery",
    "BabyCrawling",
    "BalanceBeam",
    "BandMarching",
    "BaseballPitch",
    "Basketball",
    "BasketballDunk",
    "BenchPress",
    "Biking",
    "Billiards",
    "BlowDryHair",
    "BlowingCandles",
    "BodyWeightSquats",
    "Bowling",
    "BoxingPunchingBag",
    "BoxingSpeedBag",
    "BreastStroke",
    "BrushingTeeth",
    "CleanAndJerk",
    "CliffDiving",
    "CricketBowling",
    "CricketShot",
    "CuttingInKitchen",
    "Diving",
    "Drumming",
    "Fencing",
    "FieldHockeyPenalty",
    "FloorGymnastics",
    "FrisbeeCatch",
    "FrontCrawl",
    "GolfSwing",
    "Haircut",
    "HammerThrow",
    "Hammering",
    "HandstandPushups",
    "HandstandWalking",
    "HeadMassage",
    "HighJump",
    "HorseRace",
    "HorseRiding",
    "HulaHoop",
    "IceDancing",
    "JavelinThrow",
    "JugglingBalls",
    "JumpRope",
    "JumpingJack",
    "Kayaking",
    "Knitting",
    "LongJump",
    "Lunges",
    "MilitaryParade",
    "Mixing",
    "MoppingFloor",
    "Nunchucks",
    "ParallelBars",
    "PizzaTossing",
    "PlayingCello",
    "PlayingDaf",
    "PlayingDhol",
    "PlayingFlute",
    "PlayingGuitar",
    "PlayingPiano",
    "PlayingSitar",
    "PlayingTabla",
    "PlayingViolin",
    "PoleVault",
    "PommelHorse",
    "PullUps",
    "Punch",
    "PushUps",
    "Rafting",
    "RockClimbingIndoor",
    "RopeClimbing",
    "Rowing",
    "SalsaSpin",
    "ShavingBeard",
    "Shotput",
    "SkateBoarding",
    "Skiing",
    "Skijet",
    "SkyDiving",
    "SoccerJuggling",
    "SoccerPenalty",
    "StillRings",
    "SumoWrestling",
    "Surfing",
    "Swing",
    "TableTennisShot",
    "TaiChi",
    "TennisSwing",
    "ThrowDiscus",
    "TrampolineJumping",
    "Typing",
    "UnevenBars",
    "VolleyballSpiking",
    "WalkingWithDog",
    "WallPushups",
    "WritingOnBoard",
    "YoYo",
]

HMDB51_CLASSES = [
    "brush_hair",
    "cartwheel",
    "catch",
    "chew",
    "clap",
    "climb",
    "climb_stairs",
    "dive",
    "draw_sword",
    "dribble",
    "drink",
    "eat",
    "fall_floor",
    "fencing",
    "flic_flac",
    "golf",
    "handstand",
    "hit",
    "hug",
    "jump",
    "kick",
    "kick_ball",
    "kiss",
    "laugh",
    "pick",
    "pour",
    "pullup",
    "punch",
    "push",
    "pushup",
    "ride_bike",
    "ride_horse",
    "run",
    "shake_hands",
    "shoot_ball",
    "shoot_bow",
    "shoot_gun",
    "sit",
    "situp",
    "smile",
    "smoke",
    "somersault",
    "stand",
    "swing_baseball",
    "sword",
    "sword_exercise",
    "talk",
    "throw",
    "turn",
    "walk",
    "wave",
]


_DEFAULTS = {
    "stanford40": STANFORD40_CLASSES,
    "ucf101": UCF101_CLASSES,
    "hmdb51": HMDB51_CLASSES,
}


def get_class_names(dataset: str) -> List[str]:
    """Return a copy of the canonical class list for ``dataset``."""
    try:
        return list(_DEFAULTS[dataset.lower()])
    except KeyError:
        raise ValueError(
            f"Unknown dataset '{dataset}'. Choose from: {sorted(_DEFAULTS)}"
        )


def resolve_class_names(
    checkpoint: Optional[dict], num_classes: int, fallback: Optional[List[str]]
) -> List[str]:
    """Pick class names for a loaded model.

    Priority: names stored in the checkpoint (``class_names`` key) > ``fallback``
    if its length matches ``num_classes`` > generic ``class_<i>`` labels.
    """
    if isinstance(checkpoint, dict):
        stored = checkpoint.get("class_names")
        if stored and len(stored) == num_classes:
            return list(stored)
    if fallback is not None and len(fallback) == num_classes:
        return list(fallback)
    print(
        f"Warning: no class names known for a {num_classes}-class model; "
        "using generic 'class_<i>' labels."
    )
    return [f"class_{i}" for i in range(num_classes)]
