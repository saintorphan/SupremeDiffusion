"""Shared dropdown presets for Quick Gen, Pipeline Wizard, and templates.

All preset lists use ``(label, prompt_fragment)`` tuples.  The first entry
in each list is typically a placeholder ``("-- Label --", "")``.

Both ``quick_gen.py`` and ``pipeline_wizard.py`` import from here instead
of maintaining their own copies.  ``PipelineTemplate`` can extend/override
these defaults at runtime.
"""

from __future__ import annotations

# ── Scenes / Settings ────────────────────────────────────────────────────────

SCENES = [
    ("-- Scene / Setting --", ""),
    ("Living Room", "a spacious living room with a large couch, coffee table, and warm lighting"),
    ("Bedroom", "a bedroom with a large bed, nightstands, soft ambient lighting"),
    ("Kitchen", "a modern kitchen with countertops, appliances, overhead lighting"),
    ("Office", "modern office space, clean desk, natural window light"),
    ("Bar / Club", "a dimly lit bar with neon signs, bar stools, bottles on shelves, moody atmosphere"),
    ("Bathroom", "a tiled bathroom with a bathtub, mirror, soft overhead lighting"),
    ("Hotel Room", "a hotel room with a king bed, desk, window with city view, warm lighting"),
    ("Studio / Plain", "clean studio, plain background, professional lighting"),
    ("Cafe / Restaurant", "cozy cafe interior, warm ambient lighting, modern decor"),
    ("Park / Garden", "lush green park, trees and flowers, soft natural light, peaceful setting"),
    ("Beach", "sandy beach, ocean waves in background, golden hour light"),
    ("Forest", "forest clearing, sunlight filtering through trees, natural atmosphere"),
    ("Alley", "a narrow urban alley at night, wet pavement, distant streetlights, gritty atmosphere"),
    ("Warehouse", "an abandoned warehouse with concrete floors, industrial pipes, harsh overhead lights"),
    ("Rooftop", "urban rooftop, city skyline in background, golden hour"),
    ("Mountain Landscape", "mountain vista, dramatic landscape, wide open sky"),
    ("Sci-fi / Futuristic", "futuristic setting, neon lights, holographic displays, chrome surfaces"),
    ("Vintage / Retro", "retro setting, warm vintage tones, nostalgic atmosphere"),
]

# ── Characters (Pipeline Wizard scene composer) ──────────────────────────────

CHARACTERS = [
    ("1 Woman", "one woman"),
    ("1 Man", "one man"),
    ("2 Women", "two women"),
    ("2 Men", "two men"),
    ("1 Man + 1 Woman", "one man and one woman"),
    ("3 People (Mixed)", "three people"),
    ("No People (Empty Scene)", ""),
]

# ── Body Types ────────────────────────────────────────────────────────────────

BODY_TYPES = [
    ("-- Body Type --", ""),
    ("Slim / Petite", "slim petite woman, small frame"),
    ("Average Build", "average build woman"),
    ("Curvy / Thick", "curvy thick woman, wide hips, large thighs"),
    ("Athletic / Toned", "athletic toned woman, defined muscles, fit body"),
    ("Muscular", "muscular build"),
    ("Plus Size", "plus size woman, soft round body"),
    ("Tall / Slender", "tall slender woman, long legs, lean figure"),
    ("Short / Compact", "short compact woman, small stature"),
]

# ── Skin Details ──────────────────────────────────────────────────────────────

SKIN_DETAILS = [
    ("-- Skin --", ""),
    ("Fair / Pale", "fair pale skin"),
    ("Light", "light skin, natural tone"),
    ("Medium / Olive", "medium olive skin tone"),
    ("Tan / Bronze", "tanned bronze skin, sun-kissed"),
    ("Brown", "brown skin, warm undertone"),
    ("Dark", "dark skin, rich deep tone"),
    ("Freckled", "freckled skin, light complexion"),
    ("Glowing / Dewy", "glowing dewy skin, healthy radiant complexion"),
]

# ── Hair Options ──────────────────────────────────────────────────────────────

HAIR_OPTIONS = [
    ("-- Hair --", ""),
    ("Blonde Long", "long blonde hair"),
    ("Blonde Short", "short blonde bob"),
    ("Brunette Long", "long brown hair"),
    ("Brunette Short", "short brown hair"),
    ("Black Long", "long black hair"),
    ("Black Short", "short black hair"),
    ("Red / Auburn", "red auburn hair"),
    ("Auburn", "auburn hair"),
    ("Pink", "pink hair"),
    ("White / Silver", "white silver hair"),
    ("Platinum", "platinum blonde hair"),
    ("Braided", "braided hair"),
    ("Ponytail", "hair in ponytail"),
    ("Messy / Wet", "messy wet hair, disheveled"),
]

# ── Clothing ──────────────────────────────────────────────────────────────────

CLOTHING = [
    ("-- Clothing --", ""),
    ("Casual T-Shirt", "casual t-shirt, relaxed fit"),
    ("Crop Top", "crop top, midriff exposed"),
    ("Button-Up Shirt", "button-up shirt, smart casual"),
    ("Tank Top", "tank top, casual summer look"),
    ("Sundress", "sundress, light flowing fabric"),
    ("Blazer + Jeans", "blazer over t-shirt with jeans, smart casual"),
    ("Leather Jacket", "leather jacket, edgy street style"),
    ("Evening Dress", "elegant evening dress, formal"),
    ("Hoodie", "hoodie, casual comfortable streetwear"),
    ("Sports Bra + Leggings", "sports bra and leggings, athletic wear"),
    ("Sweater", "cozy knit sweater"),
    ("Bikini", "bikini, swimwear, beach look"),
    ("Uniform", "professional uniform"),
    ("Bodysuit", "one-piece bodysuit, sleek fit"),
    ("Overalls", "denim overalls, casual workwear"),
    ("Loungewear", "wearing comfortable loungewear"),
    ("Formal Evening", "wearing formal evening attire"),
    ("Tank Top + Shorts", "wearing a tank top and shorts"),
    ("Hoodie + Sweats", "wearing a hoodie and sweatpants"),
    ("Swimwear", "wearing swimwear"),
    # Body transformation friendly — tight-fitting clothing that shows changes
    ("Tight Tank Top", "wearing a tight-fitting tank top that hugs the body"),
    ("Crop Top + Leggings", "wearing a crop top and tight leggings"),
    ("Bodycon Dress", "wearing a tight bodycon dress"),
    ("Corset + Skirt", "wearing a laced corset and skirt"),
    ("Tight Button-Up Shirt", "wearing a tight button-up shirt with visible strain"),
    ("Latex / Form-Fitting", "wearing form-fitting latex clothing"),
    ("Bikini Top + Shorts", "wearing a bikini top and cut-off shorts"),
    ("Sheer / See-Through Top", "wearing a sheer see-through top"),
]

# ── Poses ─────────────────────────────────────────────────────────────────────

POSES = [
    ("-- Pose --", ""),
    ("Standing Front", "standing facing camera, hands at sides, full body view"),
    ("Standing Profile", "standing side profile view"),
    ("Standing Arms Crossed", "standing with arms crossed, confident pose"),
    ("Sitting on Chair", "sitting on a chair, relaxed, leaning back"),
    ("Sitting on Floor", "sitting on floor, casual relaxed pose"),
    ("Lying on Back", "lying on back, relaxed pose"),
    ("Lying on Side", "lying on side, resting"),
    ("Kneeling", "kneeling upright, looking at camera"),
    ("Leaning Against Wall", "leaning back against a wall, casual"),
    ("Walking", "mid-stride walking pose, natural movement"),
    ("Hands on Hips", "standing with hands on hips, confident"),
    ("Over the Shoulder", "looking over shoulder at camera, turned away"),
    ("Sitting Cross-Legged", "sitting cross-legged on the ground"),
]

# ── Expressions ───────────────────────────────────────────────────────────────

EXPRESSIONS = [
    ("-- Expression --", ""),
    ("Smiling", "warm genuine smile, happy expression"),
    ("Laughing", "laughing joyfully, bright expression, candid moment"),
    ("Serious", "serious expression, focused gaze, composed"),
    ("Confident", "confident expression, slight smirk, self-assured"),
    ("Contemplative", "thoughtful expression, gazing into distance, pensive"),
    ("Surprised", "surprised expression, raised eyebrows, wide eyes"),
    ("Sad", "melancholic expression, downcast eyes, somber mood"),
    ("Angry", "angry expression, furrowed brow, intense stare"),
    ("Neutral", "neutral expression, calm composed face, relaxed"),
    ("Mysterious", "enigmatic expression, slight smile, knowing look"),
]

# ── Props / Accessories ───────────────────────────────────────────────────────

PROPS = [
    ("-- Props / Accessories --", ""),
    ("Sunglasses", "wearing sunglasses"),
    ("Hat / Cap", "wearing a hat"),
    ("Scarf", "wearing a scarf, draped around neck"),
    ("Bag / Purse", "carrying a bag"),
    ("Umbrella", "holding an umbrella"),
    ("Coffee Cup", "holding a coffee cup"),
    ("Book", "holding a book"),
    ("Phone", "holding a smartphone"),
    ("Headphones", "wearing headphones around neck"),
    ("None", ""),
]

# ── Camera Movements (video generation) ───────────────────────────────────────

CAMERA_MOVEMENTS = [
    ("-- Camera Movement --", ""),
    ("Static / Locked Off", "static locked-off camera, no movement"),
    ("Slow Pan Left", "slow pan left, camera gliding sideways revealing the scene"),
    ("Slow Pan Right", "slow pan right, camera gliding sideways"),
    ("Pan Left to Right", "camera panning left to right across the scene"),
    ("Pan Right to Left", "camera panning right to left across the scene"),
    ("Tilt Up", "slow tilt up, camera angling upward from feet to head"),
    ("Tilt Down", "slow tilt down, camera angling downward from head to feet"),
    ("Push In / Dolly Forward", "camera pushing in slowly, dolly forward, closing in on the subject"),
    ("Pull Back / Dolly Out", "camera pulling back slowly, dolly out, revealing the full scene"),
    ("Truck Left", "camera trucking left, moving parallel to the subject"),
    ("Truck Right", "camera trucking right, moving parallel to the subject"),
    ("Crane Up", "crane shot rising upward, revealing the scene from above"),
    ("Crane Down", "crane shot descending toward the subject"),
    ("Orbit / Arc", "camera orbiting around the subject in a slow arc"),
    ("Zoom In (Slow)", "slow zoom in, gradually tightening the frame"),
    ("Zoom In (Crash)", "crash zoom in, fast dramatic zoom punching into the subject"),
    ("Zoom Out (Reveal)", "zoom out revealing the wider scene and surroundings"),
    ("Handheld / Shaky", "handheld shaky camera, raw unstable movement, documentary feel"),
    ("Steadicam Follow", "steadicam following the subject, smooth tracking shot"),
    ("Dutch Angle", "dutch angle, tilted frame, unsettling composition"),
    ("Whip Pan", "whip pan, fast horizontal blur between two subjects"),
    ("Rack Focus", "rack focus, foreground blurs as background sharpens, then reverses"),
    ("POV / First Person", "POV shot, first person perspective, camera is the viewer's eyes"),
    ("Over-the-Shoulder", "over-the-shoulder shot, framed past the subject looking at the scene"),
    ("Low Angle", "low angle shot, camera near the floor looking up at the subject"),
    ("High Angle", "high angle shot, camera above looking down at the subject"),
    ("Bird's Eye / Top Down", "bird's eye view, camera directly overhead looking straight down"),
    ("360 Spin", "camera spinning 360 degrees around the subject"),
]

# ── Camera Angles (static starting angles for Pipeline Wizard) ────────────────

CAMERA_ANGLES = [
    ("Wide Angle (Recommended)", "wide angle establishing shot, full room visible, 24mm lens"),
    ("Medium Wide", "medium wide shot, characters and surroundings visible, 35mm lens"),
    ("Ultra Wide", "ultra wide angle panoramic shot, entire environment visible, 16mm lens"),
    ("Slightly Elevated", "slightly elevated camera angle looking down, wide shot, full scene visible"),
    ("Low Angle Wide", "low angle wide shot looking up at the scene, dramatic perspective"),
    ("Eye Level Wide", "eye level wide shot, natural perspective, full scene visible"),
]

# ── Lighting ──────────────────────────────────────────────────────────────────

LIGHTING = [
    ("Natural Daylight", "natural daylight streaming through windows, soft shadows"),
    ("Dim Evening", "dim warm evening lighting, soft shadows, cozy atmosphere"),
    ("Cinematic Dramatic", "dramatic cinematic lighting, strong key light, deep shadows"),
    ("Neon / Club", "neon colored lighting, vibrant blues and pinks, nightclub atmosphere"),
    ("Candlelight", "warm candlelight, flickering amber glow, intimate atmosphere"),
    ("Golden Hour", "golden hour sunlight, warm orange tones, long soft shadows"),
    ("Overcast Soft", "soft overcast daylight, even diffused lighting, no harsh shadows"),
    ("Harsh Overhead", "harsh overhead fluorescent lighting, clinical feel"),
    ("Backlit / Silhouette", "backlit with bright window behind, rim lighting on subjects"),
    ("Moonlight", "cool blue moonlight through a window, dark moody atmosphere"),
]

# ── Film Styles ───────────────────────────────────────────────────────────────

FILM_STYLES = [
    ("-- Film Style --", ""),
    ("Photorealistic", "photorealistic, highly detailed, 8k, professional photography"),
    ("Cinematic Film", "cinematic film still, 35mm film grain, anamorphic lens, color graded"),
    ("Magazine Editorial", "editorial magazine photography, clean sharp focus, studio quality"),
    ("Documentary", "documentary style, observational camera, natural lighting, interview framing, raw and unpolished"),
    ("Found Footage", "found footage style, shaky handheld camera, VHS tracking lines, degraded video quality, timestamp overlay, night vision green tint"),
    ("Security Camera", "security camera footage, CCTV angle, fisheye lens distortion, low resolution, grainy, timestamp in corner, black and white"),
    ("Candid / Hidden Camera", "candid hidden camera angle, voyeuristic framing, partially obstructed view, natural unposed, surveillance feel"),
    ("Home Video", "home video aesthetic, slightly overexposed, autofocus hunting, amateur framing, warm color cast, VHS artifacts"),
    ("Body Cam", "body camera footage, chest-mounted perspective, fish-eye distortion, motion blur, timestamp overlay, harsh auto-exposure"),
    ("Phone Recording", "vertical phone recording, shaky hands, auto-focus adjusting, slightly blurry, amateur framing, social media quality"),
    ("News Broadcast", "live news broadcast style, breaking news chyron, reporter framing, harsh field lighting, satellite interference"),
    ("Horror Film", "horror film cinematography, dark shadows, desaturated color, tension lighting, slow dolly, dread atmosphere"),
    ("Giallo", "giallo horror style, saturated red and blue gels, extreme close-ups, baroque framing, Italian horror aesthetic"),
    ("Grindhouse", "grindhouse film look, heavy film grain, print damage, color bleed, scratches, exploitation cinema aesthetic"),
    ("Medical / Procedure", "medical procedure recording, overhead surgical light, sterile blue-white tones, clinical framing, educational documentation style"),
    ("Slow Motion Replay", "extreme slow motion replay, 1000fps high-speed camera, every detail frozen and stretched, time almost stopped"),
    ("Noir", "film noir style, high contrast black and white, dramatic shadows, venetian blind lighting, smoke"),
    ("Neon Horror", "neon-lit horror, vibrant pink and cyan lighting, wet surfaces reflecting neon, cyberpunk horror atmosphere"),
    ("Vintage Film", "vintage 70s film photography, warm tones, slight grain, retro aesthetic"),
    ("Modern Clean", "modern clean photography, sharp focus, neutral tones, minimal"),
]

# ── Prompt Presets (default prompts for Quick Gen) ────────────────────────────

VIDEO_PROMPTS = [
    ("-- Select Prompt --", ""),
    ("Slow Orbit — Living Room",
     "A woman standing in a spacious living room, the camera slowly orbits around her, "
     "warm natural lighting, smooth steady motion, photorealistic"),
    ("Dolly Forward — Portrait",
     "A woman seated on a chair, the camera slowly pushes forward toward her face, "
     "shallow depth of field, soft lighting, cinematic"),
    ("Pan Reveal — Cityscape",
     "Camera slowly pans right revealing a city skyline at sunset, "
     "golden hour light, dramatic clouds, warm tones, photorealistic"),
    ("Crane Up — Forest",
     "Camera cranes upward from ground level through a forest canopy, "
     "dappled sunlight filtering through leaves, serene atmosphere"),
    ("Tracking Shot — Walking",
     "A person walking down a street, camera tracking alongside, "
     "steady smooth motion, urban environment, natural lighting"),
    ("Static — Conversation",
     "Two people having a conversation at a cafe table, static locked-off camera, "
     "soft ambient lighting, shallow depth of field, natural"),
]

IMAGE_PROMPTS = [
    ("-- Select Prompt --", ""),
    ("Portrait — Studio",
     "A woman posing in a professional studio, soft key light with fill, clean backdrop, "
     "shallow depth of field, photorealistic, highly detailed, 8k"),
    ("Street Photography",
     "A woman walking on a city sidewalk, candid natural moment, golden hour light, "
     "urban background with bokeh, 35mm lens, photorealistic"),
    ("Fantasy Character",
     "A warrior woman in ornate armor standing on a cliff edge, dramatic sky, "
     "volumetric lighting, epic fantasy composition, detailed textures"),
    ("Sci-fi Portrait",
     "A woman in a futuristic environment, neon lighting reflecting on her face, "
     "holographic displays in the background, cyberpunk atmosphere, cinematic"),
    ("Nature Portrait",
     "A woman standing in a sunlit meadow, wildflowers surrounding her, golden hour, "
     "warm natural tones, soft breeze in hair, photorealistic"),
    ("Dark Moody Portrait",
     "A woman in dramatic low-key lighting, single light source, deep shadows, "
     "Rembrandt lighting, emotional atmosphere, fine art photography"),
]

# ── Orbit Prompts (Pipeline Wizard) ──────────────────────────────────────────

ORBIT_PROMPTS = [
    ("Slow Orbit (Recommended)",
     "all characters are standing still, the camera slowly orbit pans around the room, "
     "smooth steady motion, consistent lighting"),
    ("Slow Pan Left to Right",
     "all characters are standing still, the camera slowly pans from left to right, "
     "revealing the room, steady smooth motion"),
    ("Slow Pan Right to Left",
     "all characters are standing still, the camera slowly pans from right to left, "
     "revealing the room, steady smooth motion"),
    ("Slow Dolly Forward",
     "all characters are standing still, the camera slowly pushes forward into the room, "
     "dolly shot, smooth steady motion"),
    ("Slow Dolly Back",
     "all characters are standing still, the camera slowly pulls back revealing the full room, "
     "dolly out, smooth steady motion"),
    ("Static (No Movement)",
     "all characters are standing still, the camera is static and locked off, "
     "subtle ambient movement only, no camera motion"),
]

# ── Realism Denoise Presets ──────────────────────────────────────────────────

REALISM_PRESETS = [
    ("Subtle (0.25)", 0.25),
    ("Light (0.35)", 0.35),
    ("Medium (0.45)", 0.45),
    ("Strong (0.55)", 0.55),
    ("Heavy (0.65)", 0.65),
    ("Full Rework (0.75)", 0.75),
]

# ── Shot Planner — camera moves (Pipeline Wizard) ────────────────────────────
# (label, prompt fragment, drift_cost 0-3)

PLANNER_CAMERA_MOVES = [
    ("Static (No Camera Move)", "the camera is static and locked off", 0),
    ("Slow Dolly Forward", "the camera slowly pushes forward", 1),
    ("Slow Dolly Back", "the camera slowly pulls back", 1),
    ("Slow Pan Left", "the camera slowly pans left", 1),
    ("Slow Pan Right", "the camera slowly pans right", 1),
    ("Slow Orbit Left", "the camera slowly orbits left around the subject", 2),
    ("Slow Orbit Right", "the camera slowly orbits right around the subject", 2),
    ("Zoom to Medium Shot", "the camera zooms in to a medium shot, waist up", 1),
    ("Zoom to Close-Up", "the camera zooms in to a close-up of the face", 2),
    ("Tilt Down", "the camera slowly tilts downward", 1),
    ("Tilt Up", "the camera slowly tilts upward", 1),
    ("Crane Up", "the camera cranes upward revealing the scene from above", 2),
]

# ── Shot Planner — subject changes (Pipeline Wizard) ─────────────────────────
# (label, prompt fragment, drift_cost 0-3)

PLANNER_SUBJECT_CHANGES = [
    ("No Change (Static Subject)", "", 0),
    ("Turns Head Slightly", "she turns her head slightly", 1),
    ("Shifts Weight", "she shifts her weight from one foot to the other", 1),
    ("Gestures with Hand", "she gestures with one hand", 1),
    ("Looks Down", "she looks down", 1),
    ("Looks at Camera", "she looks directly at the camera", 1),
    ("Body Transformation — Progressive", "", 2),  # prompt built from stages
]

# ── Drift risk thresholds ────────────────────────────────────────────────────

DRIFT_GREEN = 2    # combined cost <= this: safe
DRIFT_YELLOW = 4   # combined cost <= this: risky, suggest split
# above yellow = red: will break, auto-split required

# ── Video model options ──────────────────────────────────────────────────────

MODEL_OPTIONS_VIDEO = [
    ("Wan 2.2 Lightning (Fast)", "i2v_2_2_lightning_v2"),
    ("Wan 2.2 (Quality)", "i2v_2_2"),
    ("Wan 2.2 SVI 2 Pro (LoRAs)", "i2v_2_2_svi_2_pro"),
    ("Wan 2.2 SVI 2 Pro Enhanced Lightning v2", "i2v_2_2_svi_2_pro_lightning_v2"),
]
