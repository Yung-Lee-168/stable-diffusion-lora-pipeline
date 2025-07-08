        # 定義分類
        self.categories = {
            "Gender": ["male", "female"],
            "Age": ["child", "teenager", "young adult", "adult", "senior"],
            "Season": ["spring", "summer", "autumn", "winter"],
            "Occasion": ["casual", "formal", "business", "sport", "party", "beach", "wedding", "date", "travel", "home"],
            "Upper Body": ["t-shirt", "shirt", "jacket", "coat", "sweater", "blazer", "hoodie", "tank top", "blouse", "dress"],
            "Lower Body": ["jeans", "trousers", "shorts", "skirt", "leggings", "cargo pants", "sweatpants", "culottes", "capris", "dress"]
        }
        
        self.detailed_clothing_features = {
            "Dress Style": ["A-line dress", "sheath dress", "wrap dress", "maxi dress", "midi dress", "mini dress", "bodycon dress", "shift dress", "empire waist dress", "fit and flare dress", "slip dress", "shirt dress", "sweater dress"],
            "Shirt Features": ["button-down shirt", "polo shirt", "henley shirt", "flannel shirt", "dress shirt", "peasant blouse", "crop top", "off-shoulder top", "turtleneck", "v-neck shirt", "crew neck", "collared shirt"],
            "Jacket Types": ["denim jacket", "leather jacket", "bomber jacket", "trench coat", "peacoat", "blazer jacket", "cardigan", "windbreaker", "puffer jacket", "motorcycle jacket", "varsity jacket"],
            "Pants Details": ["skinny jeans", "straight leg jeans", "bootcut jeans", "wide leg pants", "high-waisted pants", "low-rise pants", "cropped pants", "palazzo pants", "joggers", "dress pants", "cargo pants with pockets"],
            "Skirt Varieties": ["pencil skirt", "A-line skirt", "pleated skirt", "wrap skirt", "mini skirt", "maxi skirt", "denim skirt", "leather skirt", "tulle skirt", "asymmetrical skirt"],
            "Fabric Texture": ["cotton fabric", "silk material", "denim texture", "leather finish", "wool texture", "linen fabric", "chiffon material", "velvet texture", "knit fabric", "lace material", "satin finish", "corduroy texture"],
            "Pattern Details": ["solid color", "striped pattern", "floral print", "polka dots", "geometric pattern", "animal print", "plaid pattern", "paisley design", "abstract print", "tie-dye pattern", "checkered pattern"],
            "Color Scheme": ["monochrome outfit", "pastel colors", "bright colors", "earth tones", "neutral colors", "bold colors", "metallic accents", "neon colors", "vintage colors", "gradient colors"],
            "Fit Description": ["loose fit", "tight fit", "oversized", "fitted", "relaxed fit", "tailored fit", "slim fit", "regular fit", "cropped length", "flowing silhouette", "structured shape"],
            "Style Details": ["minimalist style", "vintage style", "bohemian style", "gothic style", "preppy style", "streetwear style", "romantic style", "edgy style", "classic style", "trendy style", "elegant style"]
        }