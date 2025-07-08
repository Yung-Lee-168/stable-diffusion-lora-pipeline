# 時尚CLIP特徵值定義
# Fashion CLIP Feature Values

# 性別特徵
gender = ["male", "female", "unisex", "androgynous"]

# 年齡特徵
age = ["child", "teenager", "young adult", "adult", "middle-aged", "senior"]

# 季節特徵
season = ["spring", "summer", "autumn", "winter", "all-season"]

# 場合特徵
occasion = [
    "casual", "formal", "business", "sport", "party", "beach", 
    "wedding", "date", "travel", "home", "work", "gym", "outdoor"
]

# 上身服裝
upper_body = [
    "t-shirt", "shirt", "jacket", "coat", "sweater", "blazer", 
    "hoodie", "tank top", "blouse", "dress", "cardigan", "vest", 
    "pullover", "polo shirt", "crop top"
]

# 下身服裝
lower_body = [
    "jeans", "trousers", "shorts", "skirt", "leggings", "cargo pants", 
    "sweatpants", "culottes", "capris", "pants", "mini skirt", 
    "maxi skirt", "midi skirt", "palazzo pants"
]

# 顏色特徵
colors = [
    "black", "white", "gray", "red", "blue", "green", "yellow", 
    "pink", "purple", "orange", "brown", "navy", "beige", "cream", 
    "maroon", "olive", "teal", "coral", "lavender"
]

# 材質特徵
materials = [
    "cotton", "denim", "silk", "wool", "polyester", "leather", 
    "linen", "cashmere", "velvet", "satin", "chiffon", "lace", 
    "knit", "fleece", "canvas"
]

# 風格特徵
styles = [
    "vintage", "modern", "classic", "trendy", "bohemian", "minimalist", 
    "preppy", "gothic", "punk", "romantic", "sporty", "elegant", 
    "edgy", "chic", "retro"
]

# 圖案特徵
patterns = [
    "solid", "striped", "polka dot", "floral", "geometric", "abstract", 
    "animal print", "plaid", "checkered", "paisley", "tie-dye", 
    "gradient", "ombre"
]

# 配件特徵
accessories = [
    "hat", "scarf", "belt", "bag", "jewelry", "sunglasses", "watch", 
    "necklace", "earrings", "bracelet", "ring", "brooch", "tie", 
    "bow tie", "suspenders"
]

# 鞋類特徵
footwear = [
    "sneakers", "boots", "heels", "flats", "sandals", "loafers", 
    "pumps", "oxfords", "ballet flats", "ankle boots", "knee boots", 
    "stilettos", "wedges", "clogs"
]

# 合併所有特徵為字典（保持向後兼容）
categories = {
    "Gender": gender,
    "Age": age,
    "Season": season,
    "Occasion": occasion,
    "Upper Body": upper_body,
    "Lower Body": lower_body,
    "Colors": colors,
    "Materials": materials,
    "Styles": styles,
    "Patterns": patterns,
    "Accessories": accessories,
    "Footwear": footwear
}

# 所有特徵的扁平列表
all_features = []
for feature_list in categories.values():
    all_features.extend(feature_list)

# 去重並排序
all_features = sorted(list(set(all_features)))

print(f"✅ 特徵值.py 載入完成，共 {len(all_features)} 個特徵值")