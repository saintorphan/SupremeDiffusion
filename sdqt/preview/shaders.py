"""GLSL shader source strings for timeline effect compositing.

Vertex shader: fullscreen quad passthrough.
Fragment shaders: one per effect type, matching the ffmpeg filter builders.
"""

# -- Vertex shader (shared by all effects) --

VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 a_position;
layout(location = 1) in vec2 a_texcoord;
out vec2 v_texcoord;
void main() {
    gl_Position = vec4(a_position, 0.0, 1.0);
    v_texcoord = a_texcoord;
}
"""

# -- Passthrough (no effect, just display texture) --

PASSTHROUGH_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
void main() {
    fragColor = texture(u_texture, v_texcoord);
}
"""

# -- Hue / Saturation --

HUE_SAT_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_hue_shift;    // degrees, -180 to 180
uniform float u_saturation;   // multiplier, 0 to 3

vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0/3.0, 2.0/3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    vec3 hsv = rgb2hsv(color.rgb);
    hsv.x = fract(hsv.x + u_hue_shift / 360.0);
    hsv.y = clamp(hsv.y * u_saturation, 0.0, 1.0);
    fragColor = vec4(hsv2rgb(hsv), color.a);
}
"""

# -- Brightness / Contrast --

BRIGHTNESS_CONTRAST_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_brightness;  // -1 to 1
uniform float u_contrast;    // 0 to 3

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    vec3 c = (color.rgb - 0.5) * u_contrast + 0.5 + u_brightness;
    fragColor = vec4(clamp(c, 0.0, 1.0), color.a);
}
"""

# -- Levels --

LEVELS_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_black_point;  // 0-255
uniform float u_white_point;  // 0-255
uniform float u_gamma;        // 0.1-3.0

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    float bp = u_black_point / 255.0;
    float wp = u_white_point / 255.0;
    float range = max(wp - bp, 0.001);
    vec3 c = clamp((color.rgb - bp) / range, 0.0, 1.0);
    c = pow(c, vec3(1.0 / u_gamma));
    fragColor = vec4(c, color.a);
}
"""

# -- Shadows / Highlights --

SHADOWS_HIGHLIGHTS_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_shadows;     // -1 to 1
uniform float u_highlights;  // -1 to 1

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    float lum = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    float shadow_weight = 1.0 - smoothstep(0.0, 0.5, lum);
    float highlight_weight = smoothstep(0.5, 1.0, lum);
    vec3 c = color.rgb;
    c += u_shadows * shadow_weight * 0.5;
    c += u_highlights * highlight_weight * 0.5;
    fragColor = vec4(clamp(c, 0.0, 1.0), color.a);
}
"""

# -- Color Balance --

COLOR_BALANCE_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform vec3 u_shadow_rgb;     // -1 to 1 each
uniform vec3 u_midtone_rgb;    // -1 to 1 each
uniform vec3 u_highlight_rgb;  // -1 to 1 each

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    float lum = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    float shadow_w = 1.0 - smoothstep(0.0, 0.5, lum);
    float highlight_w = smoothstep(0.5, 1.0, lum);
    float midtone_w = 1.0 - shadow_w - highlight_w;
    midtone_w = max(midtone_w, 0.0);
    vec3 c = color.rgb;
    c += u_shadow_rgb * shadow_w;
    c += u_midtone_rgb * midtone_w;
    c += u_highlight_rgb * highlight_w;
    fragColor = vec4(clamp(c, 0.0, 1.0), color.a);
}
"""

# -- Channel Mixer --

CHANNEL_MIXER_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_rr;  // 0-3, red gain
uniform float u_gg;  // 0-3, green gain
uniform float u_bb;  // 0-3, blue gain

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    vec3 c = color.rgb * vec3(u_rr, u_gg, u_bb);
    fragColor = vec4(clamp(c, 0.0, 1.0), color.a);
}
"""

# -- Color Temperature --

COLOR_TEMPERATURE_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_temperature;  // -100 to 100
uniform float u_tint;         // -100 to 100

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    vec3 c = color.rgb;
    float t = u_temperature / 100.0;
    float ti = u_tint / 100.0;
    c.r += t * 0.1;
    c.b -= t * 0.1;
    c.g += ti * 0.1;
    fragColor = vec4(clamp(c, 0.0, 1.0), color.a);
}
"""

# -- Vibrance --

VIBRANCE_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_vibrance;  // 0 to 2

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    float avg = (color.r + color.g + color.b) / 3.0;
    float mx = max(color.r, max(color.g, color.b));
    float sat = mx - avg;
    float amt = (u_vibrance - 1.0) * (1.0 - sat) * 0.5;
    vec3 c = mix(vec3(avg), color.rgb, 1.0 + amt);
    fragColor = vec4(clamp(c, 0.0, 1.0), color.a);
}
"""

# -- Sharpen (3x3 convolution) --

SHARPEN_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_amount;  // 0 to 10
uniform vec2 u_texel_size;

void main() {
    vec4 center = texture(u_texture, v_texcoord);
    if (u_amount < 0.01) {
        fragColor = center;
        return;
    }
    vec4 sum = vec4(0.0);
    sum += texture(u_texture, v_texcoord + vec2(-1, -1) * u_texel_size);
    sum += texture(u_texture, v_texcoord + vec2( 0, -1) * u_texel_size);
    sum += texture(u_texture, v_texcoord + vec2( 1, -1) * u_texel_size);
    sum += texture(u_texture, v_texcoord + vec2(-1,  0) * u_texel_size);
    sum += texture(u_texture, v_texcoord + vec2( 1,  0) * u_texel_size);
    sum += texture(u_texture, v_texcoord + vec2(-1,  1) * u_texel_size);
    sum += texture(u_texture, v_texcoord + vec2( 0,  1) * u_texel_size);
    sum += texture(u_texture, v_texcoord + vec2( 1,  1) * u_texel_size);
    vec4 blur = sum / 8.0;
    vec4 sharp = center + (center - blur) * u_amount;
    fragColor = vec4(clamp(sharp.rgb, 0.0, 1.0), center.a);
}
"""

# -- Blur (box blur) --

BLUR_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_amount;  // 0 to 10
uniform vec2 u_texel_size;

void main() {
    if (u_amount < 0.01) {
        fragColor = texture(u_texture, v_texcoord);
        return;
    }
    int radius = int(u_amount);
    vec4 sum = vec4(0.0);
    float count = 0.0;
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            sum += texture(u_texture, v_texcoord + vec2(float(x), float(y)) * u_texel_size);
            count += 1.0;
        }
    }
    fragColor = sum / count;
}
"""

# -- Denoise (bilateral approximation via weighted blur) --

DENOISE_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_strength;  // 0 to 10
uniform vec2 u_texel_size;

void main() {
    vec4 center = texture(u_texture, v_texcoord);
    if (u_strength < 0.01) {
        fragColor = center;
        return;
    }
    int radius = max(1, int(u_strength * 0.5));
    float sigma_range = 0.1 + u_strength * 0.02;
    vec4 sum = vec4(0.0);
    float wsum = 0.0;
    for (int x = -radius; x <= radius; x++) {
        for (int y = -radius; y <= radius; y++) {
            vec4 s = texture(u_texture, v_texcoord + vec2(float(x), float(y)) * u_texel_size);
            float diff = length(s.rgb - center.rgb);
            float w = exp(-diff * diff / (2.0 * sigma_range * sigma_range));
            sum += s * w;
            wsum += w;
        }
    }
    fragColor = vec4((sum / wsum).rgb, center.a);
}
"""

# -- Vignette --

VIGNETTE_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_intensity;  // 0 to 1

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    if (u_intensity < 0.001) {
        fragColor = color;
        return;
    }
    vec2 uv = v_texcoord - 0.5;
    float d = length(uv) * 1.414;  // normalize to [0,1] at corners
    float vign = 1.0 - u_intensity * d * d;
    fragColor = vec4(color.rgb * vign, color.a);
}
"""

# -- Film Grain --

FILM_GRAIN_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform float u_intensity;  // 0 to 1
uniform float u_time;       // frame-based seed

float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    if (u_intensity < 0.001) {
        fragColor = color;
        return;
    }
    float noise = hash(v_texcoord * 1000.0 + vec2(u_time)) * 2.0 - 1.0;
    vec3 c = color.rgb + noise * u_intensity * 0.3;
    fragColor = vec4(clamp(c, 0.0, 1.0), color.a);
}
"""

# -- Composite (alpha-over for multi-track layering) --

COMPOSITE_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_base;     // bottom layer
uniform sampler2D u_overlay;  // top layer
uniform float u_alpha;        // overlay opacity 0-1

void main() {
    vec4 base = texture(u_base, v_texcoord);
    vec4 over = texture(u_overlay, v_texcoord);
    float a = over.a * u_alpha;
    fragColor = vec4(mix(base.rgb, over.rgb, a), max(base.a, a));
}
"""

# -- Crop / Zoom / Fit --

CROP_ZOOM_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform vec4 u_crop_rect;  // x, y, w, h as fractions of source (y=0 is top)

void main() {
    // Flip Y: crop rect uses y=0=top, but GL texcoords use y=0=bottom
    float flipped_y = 1.0 - u_crop_rect.y - u_crop_rect.w;
    vec2 uv = vec2(u_crop_rect.x, flipped_y) + v_texcoord * u_crop_rect.zw;
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    } else {
        fragColor = texture(u_texture, uv);
    }
}
"""

# -- 3D LUT --

LUT3D_FRAG = """
#version 330 core
in vec2 v_texcoord;
out vec4 fragColor;
uniform sampler2D u_texture;
uniform sampler3D u_lut3d;
uniform float u_lut3d_strength;  // 0.0 = bypass, 1.0 = full

void main() {
    vec4 color = texture(u_texture, v_texcoord);
    // Half-texel offset for accurate centre sampling
    float size = float(textureSize(u_lut3d, 0).x);
    float scale = (size - 1.0) / size;
    float offset = 0.5 / size;
    vec3 lut_coord = clamp(color.rgb, 0.0, 1.0) * scale + offset;
    vec3 graded = texture(u_lut3d, lut_coord).rgb;
    fragColor = vec4(mix(color.rgb, graded, u_lut3d_strength), color.a);
}
"""

# -- Registry mapping effect type -> shader source --

EFFECT_SHADERS: dict[str, str] = {
    "hue_sat": HUE_SAT_FRAG,
    "brightness_contrast": BRIGHTNESS_CONTRAST_FRAG,
    "levels": LEVELS_FRAG,
    "shadows_highlights": SHADOWS_HIGHLIGHTS_FRAG,
    "color_balance": COLOR_BALANCE_FRAG,
    "channel_mixer": CHANNEL_MIXER_FRAG,
    "color_temperature": COLOR_TEMPERATURE_FRAG,
    "vibrance": VIBRANCE_FRAG,
    "sharpen": SHARPEN_FRAG,
    "blur": BLUR_FRAG,
    "denoise": DENOISE_FRAG,
    "vignette": VIGNETTE_FRAG,
    "film_grain": FILM_GRAIN_FRAG,
    "crop_zoom": CROP_ZOOM_FRAG,
    "lut3d": LUT3D_FRAG,
}
