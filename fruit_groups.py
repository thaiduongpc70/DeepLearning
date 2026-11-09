"""Gộp 64 lớp chi tiết thành các nhóm cấp loại quả/rau để dễ nhận diện hơn.

Mỗi nhóm có tên tiếng Anh dùng làm prompt cho CLIP, tên hiển thị, và danh sách lớp thành viên.
"""

import numpy as np

# key: (tên hiển thị, prompt cho CLIP, [lớp thành viên])
GROUPS = {
    "apple": ("Apple (Táo)", "apple", ["apple_fuji", "apple_golden", "apple_granny", "apple_ligol", "apple_red_delicious", "apple_spartan"]),
    "apricot": ("Apricot (Mơ)", "apricot", ["apricot_armenian", "apricot_jerdel"]),
    "avocado": ("Avocado (Bơ)", "avocado", ["avocado_hass", "avocado_zutano"]),
    "banana": ("Banana (Chuối)", "banana", ["banana_mini", "banana_yellow"]),
    "beet": ("Beet (Củ dền)", "beetroot", ["beet_round"]),
    "broccoli": ("Broccoli (Súp lơ xanh)", "broccoli", ["cabbage_broccoli"]),
    "cauliflower": ("Cauliflower (Súp lơ trắng)", "cauliflower", ["cabbage_cauliflower"]),
    "napa_cabbage": ("Napa cabbage (Cải thảo)", "napa cabbage", ["cabbage_peking"]),
    "cabbage": ("Cabbage (Bắp cải)", "white cabbage", ["cabbage_white"]),
    "carrot": ("Carrot (Cà rốt)", "carrot", ["carrot_carrot"]),
    "corn": ("Corn (Ngô)", "corn on the cob", ["corn_corn"]),
    "cucumber": ("Cucumber (Dưa leo)", "cucumber", ["cucumber_long", "cucumber_medium"]),
    "daikon": ("Daikon (Củ cải trắng)", "daikon radish", ["daikon_daikon"]),
    "garlic": ("Garlic (Tỏi)", "garlic", ["garlic_garlic"]),
    "grape": ("Grape (Nho)", "grapes", ["grape_black", "grape_green", "grape_pink"]),
    "kiwi": ("Kiwi", "kiwi fruit", ["kiwi_kiwi"]),
    "lemon": ("Lemon (Chanh vàng)", "lemon", ["lemon_orange", "lemon_yellow"]),
    "lime": ("Lime (Chanh xanh)", "lime", ["lime_lime"]),
    "mango": ("Mango (Xoài)", "mango", ["mango_mango"]),
    "melon": ("Melon (Dưa lê)", "melon", ["melon_torpeda"]),
    "nectarine": ("Nectarine (Đào trơn)", "nectarine", ["nectarine_flat", "nectarine_round"]),
    "onion": ("Onion (Hành tây)", "onion", ["onion_purple", "onion_white"]),
    "orange": ("Orange (Cam)", "orange fruit", ["orange_orange"]),
    "pear": ("Pear (Lê)", "pear", ["pear_conference", "pear_williams_rouge"]),
    "chili": ("Chili pepper (Ớt)", "chili pepper", ["pepper_chili"]),
    "bell_pepper": ("Bell pepper (Ớt chuông)", "bell pepper", ["pepper_sweet_elonged", "pepper_sweet_green", "pepper_sweet_red", "pepper_sweet_yellow"]),
    "plum": ("Plum (Mận)", "plum", ["plum_pink", "plum_purple"]),
    "pomegranate": ("Pomegranate (Lựu)", "pomegranate", ["pomegranate_pomegranate"]),
    "pomelo": ("Pomelo (Bưởi)", "pomelo", ["pomelo_pomelo"]),
    "potato": ("Potato (Khoai tây)", "potato", ["potato_pink", "potato_yellow"]),
    "pumpkin": ("Pumpkin (Bí đỏ)", "pumpkin", ["pumpkin_elonged"]),
    "radish": ("Radish (Củ cải đỏ)", "red radish", ["raddish_elonged", "raddish_round"]),
    "lettuce": ("Lettuce (Xà lách)", "lettuce", ["salad_green"]),
    "tangerine": ("Tangerine (Quýt)", "tangerine", ["tangerine_tangerine"]),
    "tomato": ("Tomato (Cà chua)", "tomato", ["tomato_bull_heart", "tomato_cherry_red", "tomato_cherry_yellow", "tomato_pink", "tomato_slivka", "tomato_yellow"]),
    "watermelon": ("Watermelon (Dưa hấu)", "watermelon", ["watermelon_elonged"]),
    "zucchini": ("Zucchini (Bí ngòi)", "zucchini", ["zucchini_green", "zucchini_white"]),
    # Nhóm bổ sung từ dataset Kaggle (không có lớp tương ứng trong dataset đóng gói cũ)
    "pineapple": ("Pineapple (Dứa)", "pineapple", []),
    "eggplant": ("Eggplant (Cà tím)", "eggplant", []),
    "ginger": ("Ginger (Gừng)", "ginger root", []),
    "sweet_potato": ("Sweet potato (Khoai lang)", "sweet potato", []),
    "spinach": ("Spinach (Rau bina)", "spinach leaves", []),
    "turnip": ("Turnip (Củ cải turnip)", "turnip", []),
    "peas": ("Peas (Đậu Hà Lan)", "green peas", []),
}

GROUP_KEYS = list(GROUPS)
GROUP_DISPLAY = [GROUPS[k][0] for k in GROUP_KEYS]
GROUP_PROMPTS = [GROUPS[k][1] for k in GROUP_KEYS]


def class_to_group_index(class_names):
    """Trả về mảng độ dài len(class_names): chỉ số nhóm của từng lớp. Báo lỗi nếu thiếu lớp."""
    lookup = {}
    for gi, key in enumerate(GROUP_KEYS):
        for member in GROUPS[key][2]:
            lookup[member] = gi
    missing = [str(c) for c in class_names if str(c) not in lookup]
    if missing:
        raise KeyError(f"Lớp chưa có trong nhóm: {missing}")
    return np.array([lookup[str(c)] for c in class_names], dtype=np.int64)


def group_probabilities(class_probs, class_names):
    """Cộng xác suất các lớp thành viên để ra xác suất theo nhóm."""
    index = class_to_group_index(class_names)
    out = np.zeros(len(GROUP_KEYS), dtype=np.float64)
    np.add.at(out, index, np.asarray(class_probs, dtype=np.float64))
    return out
