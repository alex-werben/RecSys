from dataclasses import dataclass, field

@dataclass
class InteractionsColumnParams:
    user_id: str = field(default="user_id")
    item_id: str = field(default="item_id")
    weight: str = field(default="weight")
    datetime: str = field(default="datetime")
