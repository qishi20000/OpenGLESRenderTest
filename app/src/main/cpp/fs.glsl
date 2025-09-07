/////////////////////////////// Source file 0/////////////////////////////
#version 410 core

/////////////////////////////// Source file 1/////////////////////////////

#extension GL_ARB_shading_language_packing : enable

#extension GL_GOOGLE_cpp_style_line_directive : enable

/////////////////////////////// Source file 2/////////////////////////////
#define SPIRV_CROSS_CONSTANT_ID_0 3
#define SPIRV_CROSS_CONSTANT_ID_1 64
#define SPIRV_CROSS_CONSTANT_ID_4 2048
#define SPIRV_CROSS_CONSTANT_ID_6 false
#define SPIRV_CROSS_CONSTANT_ID_7 false
#define SPIRV_CROSS_CONSTANT_ID_2 false
#define SPIRV_CROSS_CONSTANT_ID_5 false
#define SPIRV_CROSS_CONSTANT_ID_8 2
#define SPIRV_CROSS_CONSTANT_ID_9 3


/////////////////////////////// Source file 3/////////////////////////////

#define TARGET_GL_ENVIRONMENT
#define FILAMENT_OPENGL_SEMANTICS
#define FILAMENT_HAS_FEATURE_TEXTURE_GATHER
#define FILAMENT_HAS_FEATURE_INSTANCING
#define FILAMENT_EFFECTIVE_VERSION __VERSION__
#define VARYING in
#define SHADING_MODEL_LIT
#define FILAMENT_QUALITY_LOW    0
#define FILAMENT_QUALITY_NORMAL 1
#define FILAMENT_QUALITY_HIGH   2
#define FILAMENT_QUALITY FILAMENT_QUALITY_HIGH

precision highp float;
precision highp int;

#ifndef SPIRV_CROSS_CONSTANT_ID_0
#define SPIRV_CROSS_CONSTANT_ID_0 1
#endif
const int BACKEND_FEATURE_LEVEL = SPIRV_CROSS_CONSTANT_ID_0;

#ifndef SPIRV_CROSS_CONSTANT_ID_1
#define SPIRV_CROSS_CONSTANT_ID_1 64
#endif
const int CONFIG_MAX_INSTANCES = SPIRV_CROSS_CONSTANT_ID_1;

#ifndef SPIRV_CROSS_CONSTANT_ID_4
#define SPIRV_CROSS_CONSTANT_ID_4 1024
#endif
const int CONFIG_FROXEL_BUFFER_HEIGHT = SPIRV_CROSS_CONSTANT_ID_4;

#ifndef SPIRV_CROSS_CONSTANT_ID_6
#define SPIRV_CROSS_CONSTANT_ID_6 false
#endif
const bool CONFIG_DEBUG_DIRECTIONAL_SHADOWMAP = SPIRV_CROSS_CONSTANT_ID_6;

#ifndef SPIRV_CROSS_CONSTANT_ID_7
#define SPIRV_CROSS_CONSTANT_ID_7 false
#endif
const bool CONFIG_DEBUG_FROXEL_VISUALIZATION = SPIRV_CROSS_CONSTANT_ID_7;

#ifndef SPIRV_CROSS_CONSTANT_ID_2
#define SPIRV_CROSS_CONSTANT_ID_2 false
#endif
const bool CONFIG_STATIC_TEXTURE_TARGET_WORKAROUND = SPIRV_CROSS_CONSTANT_ID_2;

#ifndef SPIRV_CROSS_CONSTANT_ID_5
#define SPIRV_CROSS_CONSTANT_ID_5 false
#endif
const bool CONFIG_POWER_VR_SHADER_WORKAROUNDS = SPIRV_CROSS_CONSTANT_ID_5;

#ifndef SPIRV_CROSS_CONSTANT_ID_8
#define SPIRV_CROSS_CONSTANT_ID_8 2
#endif
const int CONFIG_STEREO_EYE_COUNT = SPIRV_CROSS_CONSTANT_ID_8;

#ifndef SPIRV_CROSS_CONSTANT_ID_9
#define SPIRV_CROSS_CONSTANT_ID_9 3
#endif
const int CONFIG_SH_BANDS_COUNT = SPIRV_CROSS_CONSTANT_ID_9;

#define CONFIG_MAX_STEREOSCOPIC_EYES 4

#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                      
#endif
#if defined(FILAMENT_VULKAN_SEMANTICS)
#define LAYOUT_LOCATION(x) layout(location = x)
#else
#define LAYOUT_LOCATION(x)
#endif

#define bool2    bvec2
#define bool3    bvec3
#define bool4    bvec4

#define int2     ivec2
#define int3     ivec3
#define int4     ivec4

#define uint2    uvec2
#define uint3    uvec3
#define uint4    uvec4

#define float2   vec2
#define float3   vec3
#define float4   vec4

#define float3x3 mat3
#define float4x4 mat4

// To workaround an adreno crash (#5294), we need ensure that a method with
// parameter 'const mat4' does not call another method also with a 'const mat4'
// parameter (i.e. mulMat4x4Float3). So we remove the const modifier for
// materials compiled for vulkan+mobile.
#if defined(TARGET_VULKAN_ENVIRONMENT) && defined(TARGET_MOBILE)
   #define highp_mat4 highp mat4
#else
   #define highp_mat4 const highp mat4
#endif

#define MATERIAL_HAS_REFLECTIONS
#define MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY

#define BLEND_MODE_OPAQUE
#define POST_LIGHTING_BLEND_MODE_TRANSPARENT

#define GEOMETRIC_SPECULAR_AA
#define SPECULAR_AMBIENT_OCCLUSION 1
#define MULTI_BOUNCE_AMBIENT_OCCLUSION 1
#define MATERIAL_HAS_BASE_COLOR
#define MATERIAL_HAS_ROUGHNESS
#define MATERIAL_HAS_METALLIC
#define MATERIAL_HAS_AMBIENT_OCCLUSION
#define MATERIAL_HAS_EMISSIVE
#define MATERIAL_HAS_SPECULAR_FACTOR
#define MATERIAL_HAS_SPECULAR_COLOR_FACTOR
#define SHADING_INTERPOLATION 

#define HAS_ATTRIBUTE_POSITION
#define HAS_ATTRIBUTE_TANGENTS
#define HAS_ATTRIBUTE_UV0

#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                
#endif
//------------------------------------------------------------------------------
// Varyings
//------------------------------------------------------------------------------

LAYOUT_LOCATION(4) VARYING highp vec4 vertex_worldPosition;

#if defined(HAS_ATTRIBUTE_TANGENTS)
LAYOUT_LOCATION(5) SHADING_INTERPOLATION VARYING mediump vec3 vertex_worldNormal;
#if defined(MATERIAL_NEEDS_TBN)
LAYOUT_LOCATION(6) SHADING_INTERPOLATION VARYING mediump vec4 vertex_worldTangent;
#endif
#endif

LAYOUT_LOCATION(7) VARYING highp vec4 vertex_position;

#if defined(FILAMENT_HAS_FEATURE_INSTANCING)
LAYOUT_LOCATION(8) flat VARYING highp int instance_index;
highp int logical_instance_index;
#endif

#if defined(HAS_ATTRIBUTE_COLOR)
LAYOUT_LOCATION(9) VARYING mediump vec4 vertex_color;
#endif

#if defined(HAS_ATTRIBUTE_UV0) && !defined(HAS_ATTRIBUTE_UV1)
LAYOUT_LOCATION(10) VARYING highp vec2 vertex_uv01;
#elif defined(HAS_ATTRIBUTE_UV1)
LAYOUT_LOCATION(10) VARYING highp vec4 vertex_uv01;
#endif

// Note that fragColor is an output and is not declared here; see main.fs and depth_main.fs

#if defined(VARIANT_HAS_STEREO) && defined(FILAMENT_STEREO_INSTANCED)
#if defined(GL_ES) && defined(FILAMENT_GLSLANG)
// On ES, gl_ClipDistance is not a built-in, so we have to rely on EXT_clip_cull_distance
// However, this extension is not supported by glslang, so we instead write to
// filament_gl_ClipDistance, which will get decorated at the SPIR-V stage to refer to the built-in.
// The location here does not matter, so long as it doesn't conflict with others.
LAYOUT_LOCATION(100) out float filament_gl_ClipDistance[2];
#define FILAMENT_CLIPDISTANCE filament_gl_ClipDistance
#else
// If we're on Desktop GL (or not running shaders through glslang), we're free to use gl_ClipDistance
#define FILAMENT_CLIPDISTANCE gl_ClipDistance
#endif // GL_ES && FILAMENT_GLSLANG
#endif // VARIANT_HAS_STEREO


#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                    
#endif


#if defined(VARIANT_HAS_SKINNING_OR_MORPHING)
struct BoneData {
    highp mat3x4 transform;    // bone transform is mat4x3 stored in row-major (last row [0,0,0,1])
    highp float3 cof0;         // 3 first cofactor matrix of transform's upper left
    highp float cof1x;         // 4th cofactor
};
#endif

struct PerRenderableData {
    highp mat4 worldFromModelMatrix;
    highp mat3 worldFromModelNormalMatrix;
    highp int morphTargetCount;
    highp int flagsChannels;                   // see packFlags() below (0x00000fll)
    highp int objectId;                        // used for picking
    highp float userData;   // TODO: We need a better solution, this currently holds the average local scale for the renderable
    highp vec4 reserved[8];
};

// Bits for flagsChannels
#define FILAMENT_OBJECT_SKINNING_ENABLED_BIT   0x100
#define FILAMENT_OBJECT_MORPHING_ENABLED_BIT   0x200
#define FILAMENT_OBJECT_CONTACT_SHADOWS_BIT    0x400
#define FILAMENT_OBJECT_INSTANCE_BUFFER_BIT    0x800

layout(std140) uniform FrameUniforms {
    mat4 viewFromWorldMatrix;
    mat4 worldFromViewMatrix;
    mat4 clipFromViewMatrix;
    mat4 viewFromClipMatrix;
    mat4 clipFromWorldMatrix[4];
    mat4 worldFromClipMatrix;
    mat4 userWorldFromWorldMatrix;
    vec4 clipTransform;
    vec2 clipControl;
    float time;
    float temporalNoise;
    vec4 userTime;
    vec4 resolution;
    vec2 logicalViewportScale;
    vec2 logicalViewportOffset;
    float lodBias;
    float refractionLodOffset;
    lowp vec2 derivativesScale;
    float oneOverFarMinusNear;
    float nearOverFarMinusNear;
    float cameraFar;
    float exposure;
    float ev100;
    float needsAlphaChannel;
    lowp float aoSamplingQualityAndEdgeDistance;
    lowp float aoBentNormals;
    lowp vec4 zParams;
    lowp uvec3 fParams;
    lowp int lightChannels;
    lowp vec2 froxelCountXY;
    float iblLuminance;
    float iblRoughnessOneLevel;
    lowp vec3 iblSH[9];
    vec3 lightDirection;
    lowp float padding0;
    vec4 lightColorIntensity;
    vec4 sun;
    vec2 shadowFarAttenuationParams;
    lowp int directionalShadows;
    lowp float ssContactShadowDistance;
    vec4 cascadeSplits;
    lowp int cascades;
    lowp float shadowPenumbraRatioScale;
    vec2 lightFarAttenuationParams;
    lowp float vsmExponent;
    lowp float vsmDepthScale;
    lowp float vsmLightBleedReduction;
    lowp uint shadowSamplingType;
    vec3 fogDensity;
    float fogStart;
    float fogMaxOpacity;
    uint fogMinMaxMip;
    float fogHeightFalloff;
    float fogCutOffDistance;
    vec3 fogColor;
    float fogColorFromIbl;
    float fogInscatteringStart;
    float fogInscatteringSize;
    float fogOneOverFarMinusNear;
    float fogNearOverFarMinusNear;
    mat3 fogFromWorldMatrix;
    mat4 ssrReprojection;
    mat4 ssrUvFromViewMatrix;
    lowp float ssrThickness;
    lowp float ssrBias;
    lowp float ssrDistance;
    lowp float ssrStride;
    vec4 custom[4];
    int rec709;
    lowp float es2Reserved0;
    lowp float es2Reserved1;
    lowp float es2Reserved2;
    lowp vec4 reserved[40];
} frameUniforms;

layout(std140) uniform ObjectUniforms {
    PerRenderableData data[CONFIG_MAX_INSTANCES];
} objectUniforms;

layout(std140) uniform MaterialParams {
    vec4 baseColorFactor;
    float metallicFactor;
    float roughnessFactor;
    float normalScale;
    float aoStrength;
    vec3 emissiveFactor;
    float emissiveStrength;
    float specularStrength;
    vec3 specularColorFactor;
    lowp float _specularAntiAliasingVariance;
    lowp float _specularAntiAliasingThreshold;
    bool _doubleSided;
} materialParams;

uniform mediump sampler2DArrayShadow sampler0_shadowMap;
uniform mediump sampler2D sampler0_iblDFG;
uniform mediump samplerCube sampler0_iblSpecular;
uniform mediump sampler2DArray sampler0_ssao;
uniform mediump sampler2DArray sampler0_ssr;
uniform highp sampler2D sampler0_structure;
uniform mediump samplerCube sampler0_fog;

uniform  sampler2D materialParams_metallicRoughnessMap;
uniform  sampler2D materialParams_occlusionMap;

#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                   
#endif
//------------------------------------------------------------------------------
// Common math
//------------------------------------------------------------------------------

/** @public-api */
#define PI                 3.14159265359
/** @public-api */
#define HALF_PI            1.570796327

#define MEDIUMP_FLT_MAX    65504.0
#define MEDIUMP_FLT_MIN    0.00006103515625

#ifdef TARGET_MOBILE
#define FLT_EPS            MEDIUMP_FLT_MIN
#define saturateMediump(x) min(x, MEDIUMP_FLT_MAX)
#else
#define FLT_EPS            1e-5
#define saturateMediump(x) x
#endif

#define saturate(x)        clamp(x, 0.0, 1.0)

//------------------------------------------------------------------------------
// Scalar operations
//------------------------------------------------------------------------------

/**
 * Computes x^5 using only multiply operations.
 *
 * @public-api
 */
float pow5(float x) {
    float x2 = x * x;
    return x2 * x2 * x;
}

/**
 * Computes x^2 as a single multiplication.
 *
 * @public-api
 */
float sq(float x) {
    return x * x;
}

//------------------------------------------------------------------------------
// Vector operations
//------------------------------------------------------------------------------

/**
 * Returns the maximum component of the specified vector.
 *
 * @public-api
 */
float max3(const vec3 v) {
    return max(v.x, max(v.y, v.z));
}

float vmax(const vec2 v) {
    return max(v.x, v.y);
}

float vmax(const vec3 v) {
    return max(v.x, max(v.y, v.z));
}

float vmax(const vec4 v) {
    return max(max(v.x, v.y), max(v.y, v.z));
}

/**
 * Returns the minimum component of the specified vector.
 *
 * @public-api
 */
float min3(const vec3 v) {
    return min(v.x, min(v.y, v.z));
}

float vmin(const vec2 v) {
    return min(v.x, v.y);
}

float vmin(const vec3 v) {
    return min(v.x, min(v.y, v.z));
}

float vmin(const vec4 v) {
    return min(min(v.x, v.y), min(v.y, v.z));
}

//------------------------------------------------------------------------------
// Trigonometry
//------------------------------------------------------------------------------

/**
 * Approximates acos(x) with a max absolute error of 9.0x10^-3.
 * Valid in the range -1..1.
 */
float acosFast(float x) {
    // Lagarde 2014, "Inverse trigonometric functions GPU optimization for AMD GCN architecture"
    // This is the approximation of degree 1, with a max absolute error of 9.0x10^-3
    float y = abs(x);
    float p = -0.1565827 * y + 1.570796;
    p *= sqrt(1.0 - y);
    return x >= 0.0 ? p : PI - p;
}

/**
 * Approximates acos(x) with a max absolute error of 9.0x10^-3.
 * Valid only in the range 0..1.
 */
float acosFastPositive(float x) {
    float p = -0.1565827 * x + 1.570796;
    return p * sqrt(1.0 - x);
}

//------------------------------------------------------------------------------
// Matrix and quaternion operations
//------------------------------------------------------------------------------

/**
 * Multiplies the specified 3-component vector by the 4x4 matrix (m * v) in
 * high precision.
 *
 * @public-api
 */
highp vec4 mulMat4x4Float3(const highp mat4 m, const highp vec3 v) {
    return v.x * m[0] + (v.y * m[1] + (v.z * m[2] + m[3]));
}

/**
 * Multiplies the specified 3-component vector by the 3x3 matrix (m * v) in
 * high precision.
 *
 * @public-api
 */
highp vec3 mulMat3x3Float3(const highp mat4 m, const highp vec3 v) {
    return v.x * m[0].xyz + (v.y * m[1].xyz + (v.z * m[2].xyz));
}

/**
 * Extracts the normal vector of the tangent frame encoded in the specified quaternion.
 */
void toTangentFrame(const highp vec4 q, out highp vec3 n) {
    n = vec3( 0.0,  0.0,  1.0) +
        vec3( 2.0, -2.0, -2.0) * q.x * q.zwx +
        vec3( 2.0,  2.0, -2.0) * q.y * q.wzy;
}

/**
 * Extracts the normal and tangent vectors of the tangent frame encoded in the
 * specified quaternion.
 */
void toTangentFrame(const highp vec4 q, out highp vec3 n, out highp vec3 t) {
    toTangentFrame(q, n);
    t = vec3( 1.0,  0.0,  0.0) +
        vec3(-2.0,  2.0, -2.0) * q.y * q.yxw +
        vec3(-2.0,  2.0,  2.0) * q.z * q.zwx;
}

highp mat3 cofactor(const highp mat3 m) {
    highp float a = m[0][0];
    highp float b = m[1][0];
    highp float c = m[2][0];
    highp float d = m[0][1];
    highp float e = m[1][1];
    highp float f = m[2][1];
    highp float g = m[0][2];
    highp float h = m[1][2];
    highp float i = m[2][2];

    highp mat3 cof;
    cof[0][0] = e * i - f * h;
    cof[0][1] = c * h - b * i;
    cof[0][2] = b * f - c * e;
    cof[1][0] = f * g - d * i;
    cof[1][1] = a * i - c * g;
    cof[1][2] = c * d - a * f;
    cof[2][0] = d * h - e * g;
    cof[2][1] = b * g - a * h;
    cof[2][2] = a * e - b * d;
    return cof;
}

//------------------------------------------------------------------------------
// Random
//------------------------------------------------------------------------------

/*
 * Random number between 0 and 1, using interleaved gradient noise.
 * w must not be normalized (e.g. window coordinates)
 */
float interleavedGradientNoise(highp vec2 w) {
    const vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
    return fract(m.z * fract(dot(w, m.xy)));
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                         
#endif
//------------------------------------------------------------------------------
// Instancing
// ------------------------------------------------------------------------------------------------

highp mat4 object_uniforms_worldFromModelMatrix;
highp mat3 object_uniforms_worldFromModelNormalMatrix;
highp int object_uniforms_morphTargetCount;
highp int object_uniforms_flagsChannels;                   // see packFlags() below (0x00000fll)
highp int object_uniforms_objectId;                        // used for picking
highp float object_uniforms_userData;   // TODO: We need a better solution, this currently holds the average local scale for the renderable

//------------------------------------------------------------------------------
// Instance access
//------------------------------------------------------------------------------

void initObjectUniforms() {
    // Adreno drivers workarounds:
    // - We need to copy each field separately because non-const array access in a UBO fails
    //    e.g.: this fails `p = objectUniforms.data[instance_index];`
    // - We can't use a struct to hold the result because Adreno driver ignore precision qualifiers
    //   on fields of structs, unless they're in a UBO (which we just copied out of).

#if defined(FILAMENT_HAS_FEATURE_INSTANCING)
    highp int i;
#   if defined(MATERIAL_HAS_INSTANCES)
    // instancing handled by the material
    if ((objectUniforms.data[0].flagsChannels & FILAMENT_OBJECT_INSTANCE_BUFFER_BIT) != 0) {
        // hybrid instancing, we have a instance buffer per object
        i = logical_instance_index;
    } else {
        // fully manual instancing
        i = 0;
    }
#   else
    // automatic instancing
    i = logical_instance_index;
#   endif
#else
    // we don't support instancing (e.g. ES2)
    const int i = 0;
#endif
    object_uniforms_worldFromModelMatrix        = objectUniforms.data[i].worldFromModelMatrix;
    object_uniforms_worldFromModelNormalMatrix  = objectUniforms.data[i].worldFromModelNormalMatrix;
    object_uniforms_morphTargetCount            = objectUniforms.data[i].morphTargetCount;
    object_uniforms_flagsChannels               = objectUniforms.data[i].flagsChannels;
    object_uniforms_objectId                    = objectUniforms.data[i].objectId;
    object_uniforms_userData                    = objectUniforms.data[i].userData;
}

#if defined(FILAMENT_HAS_FEATURE_INSTANCING) && defined(MATERIAL_HAS_INSTANCES)
/** @public-api */
highp int getInstanceIndex() {
    return logical_instance_index;
}
#endif
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                        
#endif
//------------------------------------------------------------------------------
// Shadowing
//------------------------------------------------------------------------------


#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                    
#endif
// These variables should be in a struct but some GPU drivers ignore the
// precision qualifier on individual struct members
highp mat3  shading_tangentToWorld;   // TBN matrix
highp vec3  shading_position;         // position of the fragment in world space
      vec3  shading_view;             // normalized vector from the fragment to the eye
      vec3  shading_normal;           // normalized transformed normal, in world space
      vec3  shading_geometricNormal;  // normalized geometric normal, in world space
      vec3  shading_reflected;        // reflection of view about normal
      float shading_NoV;              // dot(normal, view), always strictly >= MIN_N_DOT_V

#if defined(MATERIAL_HAS_BENT_NORMAL)
      vec3  shading_bentNormal;       // normalized transformed normal, in world space
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT)
      vec3  shading_clearCoatNormal;  // normalized clear coat layer normal, in world space
#endif

highp vec2 shading_normalizedViewportCoord;
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                     
#endif
//------------------------------------------------------------------------------
// Common color operations
//------------------------------------------------------------------------------

/**
 * Computes the luminance of the specified linear RGB color using the
 * luminance coefficients from Rec. 709.
 *
 * @public-api
 */
float luminance(const vec3 linear) {
    return dot(linear, vec3(0.2126, 0.7152, 0.0722));
}

/**
 * Computes the pre-exposed intensity using the specified intensity and exposure.
 * This function exists to force highp precision on the two parameters
 */
float computePreExposedIntensity(const highp float intensity, const highp float exposure) {
    return intensity * exposure;
}

void unpremultiply(inout vec4 color) {
    color.rgb /= max(color.a, FLT_EPS);
}

/**
 * Applies a full range YCbCr to sRGB conversion and returns an RGB color.
 *
 * @public-api
 */
vec3 ycbcrToRgb(float luminance, vec2 cbcr) {
    // Taken from https://developer.apple.com/documentation/arkit/arframe/2867984-capturedimage
    const mat4 ycbcrToRgbTransform = mat4(
         1.0000,  1.0000,  1.0000,  0.0000,
         0.0000, -0.3441,  1.7720,  0.0000,
         1.4020, -0.7141,  0.0000,  0.0000,
        -0.7010,  0.5291, -0.8860,  1.0000
    );
    return (ycbcrToRgbTransform * vec4(luminance, cbcr, 1.0)).rgb;
}

//------------------------------------------------------------------------------
// Tone mapping operations
//------------------------------------------------------------------------------

/*
 * The input must be in the [0, 1] range.
 */
vec3 Inverse_Tonemap_Filmic(const vec3 x) {
    return (0.03 - 0.59 * x - sqrt(0.0009 + 1.3702 * x - 1.0127 * x * x)) / (-5.02 + 4.86 * x);
}

/**
 * Applies the inverse of the tone mapping operator to the specified HDR or LDR
 * sRGB (non-linear) color and returns a linear sRGB color. The inverse tone mapping
 * operator may be an approximation of the real inverse operation.
 *
 * @public-api
 */
vec3 inverseTonemapSRGB(vec3 color) {
    // sRGB input
    color = clamp(color, 0.0, 1.0);
    return Inverse_Tonemap_Filmic(pow(color, vec3(2.2)));
}

/**
 * Applies the inverse of the tone mapping operator to the specified HDR or LDR
 * linear RGB color and returns a linear RGB color. The inverse tone mapping operator
 * may be an approximation of the real inverse operation.
 *
 * @public-api
 */
vec3 inverseTonemap(vec3 linear) {
    // Linear input
    return Inverse_Tonemap_Filmic(clamp(linear, 0.0, 1.0));
}

//------------------------------------------------------------------------------
// Common texture operations
//------------------------------------------------------------------------------

/**
 * Decodes the specified RGBM value to linear HDR RGB.
 */
vec3 decodeRGBM(vec4 c) {
    c.rgb *= (c.a * 16.0);
    return c.rgb * c.rgb;
}

//------------------------------------------------------------------------------
// Common screen-space operations
//------------------------------------------------------------------------------

// returns the frag coord in the GL convention with (0, 0) at the bottom-left
// resolution : width, height
highp vec2 getFragCoord(const highp vec2 resolution) {
#if defined(TARGET_METAL_ENVIRONMENT) || defined(TARGET_VULKAN_ENVIRONMENT)
    return vec2(gl_FragCoord.x, resolution.y - gl_FragCoord.y);
#else
    return gl_FragCoord.xy;
#endif
}

//------------------------------------------------------------------------------
// Common debug
//------------------------------------------------------------------------------

vec3 heatmap(float v) {
    vec3 r = v * 2.1 - vec3(1.8, 1.14, 0.3);
    return 1.0 - r * r;
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                     
#endif
#if defined(TARGET_MOBILE)
    // min roughness such that (MIN_PERCEPTUAL_ROUGHNESS^4) > 0 in fp16 (i.e. 2^(-14/4), rounded up)
    #define MIN_PERCEPTUAL_ROUGHNESS 0.089
    #define MIN_ROUGHNESS            0.007921
#else
    #define MIN_PERCEPTUAL_ROUGHNESS 0.045
    #define MIN_ROUGHNESS            0.002025
#endif

#define MIN_N_DOT_V 1e-4

float clampNoV(float NoV) {
    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    return max(NoV, MIN_N_DOT_V);
}

vec3 computeDiffuseColor(const vec4 baseColor, float metallic) {
    return baseColor.rgb * (1.0 - metallic);
}

vec3 computeF0(const vec4 baseColor, float metallic, float reflectance) {
    return baseColor.rgb * metallic + (reflectance * (1.0 - metallic));
}

float computeDielectricF0(float reflectance) {
    return 0.16 * reflectance * reflectance;
}

float computeMetallicFromSpecularColor(const vec3 specularColor) {
    return max3(specularColor);
}

float computeRoughnessFromGlossiness(float glossiness) {
    return 1.0 - glossiness;
}

float perceptualRoughnessToRoughness(float perceptualRoughness) {
    return perceptualRoughness * perceptualRoughness;
}

float roughnessToPerceptualRoughness(float roughness) {
    return sqrt(roughness);
}

float iorToF0(float transmittedIor, float incidentIor) {
    return sq((transmittedIor - incidentIor) / (transmittedIor + incidentIor));
}

float f0ToIor(float f0) {
    float r = sqrt(f0);
    return (1.0 + r) / (1.0 - r);
}

vec3 f0ClearCoatToSurface(const vec3 f0) {
    // Approximation of iorTof0(f0ToIor(f0), 1.5)
    // This assumes that the clear coat layer has an IOR of 1.5
#if FILAMENT_QUALITY == FILAMENT_QUALITY_LOW
    return saturate(f0 * (f0 * 0.526868 + 0.529324) - 0.0482256);
#else
    return saturate(f0 * (f0 * (0.941892 - 0.263008 * f0) + 0.346479) - 0.0285998);
#endif
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                      
#endif
//------------------------------------------------------------------------------
// Uniforms access
//------------------------------------------------------------------------------

/** @public-api */
highp mat4 getViewFromWorldMatrix() {
    return frameUniforms.viewFromWorldMatrix;
}

/** @public-api */
highp mat4 getWorldFromViewMatrix() {
    return frameUniforms.worldFromViewMatrix;
}

/** @public-api */
highp mat4 getClipFromViewMatrix() {
    return frameUniforms.clipFromViewMatrix;
}

/** @public-api */
highp mat4 getViewFromClipMatrix() {
    return frameUniforms.viewFromClipMatrix;
}

/** @public-api */
highp mat4 getClipFromWorldMatrix() {
#if defined(VARIANT_HAS_STEREO) && defined(FILAMENT_STEREO_INSTANCED)
    int eye = instance_index % CONFIG_STEREO_EYE_COUNT;
    return frameUniforms.clipFromWorldMatrix[eye];
#elif defined(VARIANT_HAS_STEREO) && defined(FILAMENT_STEREO_MULTIVIEW)
    return frameUniforms.clipFromWorldMatrix[gl_ViewID_OVR];
#else
    return frameUniforms.clipFromWorldMatrix[0];
#endif
}

/** @public-api */
highp mat4 getWorldFromClipMatrix() {
    return frameUniforms.worldFromClipMatrix;
}

/** @public-api */
highp mat4 getUserWorldFromWorldMatrix() {
    return frameUniforms.userWorldFromWorldMatrix;
}

/** @public-api */
float getTime() {
    return frameUniforms.time;
}

/** @public-api */
highp vec4 getUserTime() {
    return frameUniforms.userTime;
}

/** @public-api **/
highp float getUserTimeMod(float m) {
    return mod(mod(frameUniforms.userTime.x, m) + mod(frameUniforms.userTime.y, m), m);
}

/**
 * Transforms a texture UV to make it suitable for a render target attachment.
 *
 * In Vulkan and Metal, texture coords are Y-down but in OpenGL they are Y-up. This wrapper function
 * accounts for these differences. When sampling from non-render targets (i.e. uploaded textures)
 * these differences do not matter because OpenGL has a second piece of backwardness, which is that
 * the first row of texels in glTexImage2D is interpreted as the bottom row.
 *
 * To protect users from these differences, we recommend that materials in the SURFACE domain
 * leverage this wrapper function when sampling from offscreen render targets.
 *
 * @public-api
 */
highp vec2 uvToRenderTargetUV(const highp vec2 uv) {
#if defined(TARGET_METAL_ENVIRONMENT) || defined(TARGET_VULKAN_ENVIRONMENT)
    return vec2(uv.x, 1.0 - uv.y);
#else
    return uv;
#endif
}

// TODO: below shouldn't be accessible from post-process materials

/** @public-api */
highp vec4 getResolution() {
    return frameUniforms.resolution;
}

/** @public-api */
highp vec3 getWorldCameraPosition() {
    return frameUniforms.worldFromViewMatrix[3].xyz;
}

/** @public-api, @deprecated use getUserWorldPosition() or getUserWorldFromWorldMatrix() instead  */
highp vec3 getWorldOffset() {
    return getUserWorldFromWorldMatrix()[3].xyz;
}

/** @public-api */
float getExposure() {
    // NOTE: this is a highp uniform only to work around #3602 (qualcomm)
    // We are intentionally casting it to mediump here, as per the Materials doc.
    return frameUniforms.exposure;
}

/** @public-api */
float getEV100() {
    return frameUniforms.ev100;
}

//------------------------------------------------------------------------------
// user defined globals
//------------------------------------------------------------------------------

highp vec4 getMaterialGlobal0() {
    return frameUniforms.custom[0];
}

highp vec4 getMaterialGlobal1() {
    return frameUniforms.custom[1];
}

highp vec4 getMaterialGlobal2() {
    return frameUniforms.custom[2];
}

highp vec4 getMaterialGlobal3() {
    return frameUniforms.custom[3];
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0             
#endif
//------------------------------------------------------------------------------
// Uniforms access
//------------------------------------------------------------------------------

/** sort-of public */
float getObjectUserData() {
    return object_uniforms_userData;
}

//------------------------------------------------------------------------------
// Attributes access
//------------------------------------------------------------------------------

#if defined(HAS_ATTRIBUTE_COLOR)
/** @public-api */
vec4 getColor() {
    return vertex_color;
}
#endif

#if defined(HAS_ATTRIBUTE_UV0)
/** @public-api */
highp vec2 getUV0() {
    return vertex_uv01.xy;
}
#endif

#if defined(HAS_ATTRIBUTE_UV1)
/** @public-api */
highp vec2 getUV1() {
    return vertex_uv01.zw;
}
#endif

#if defined(BLEND_MODE_MASKED)
/** @public-api */
float getMaskThreshold() {
    return materialParams._maskThreshold;
}
#endif

/** @public-api */
highp mat3 getWorldTangentFrame() {
    return shading_tangentToWorld;
}

/** @public-api */
highp vec3 getWorldPosition() {
    return shading_position;
}

/** @public-api */
highp vec3 getUserWorldPosition() {
    return mulMat4x4Float3(getUserWorldFromWorldMatrix(), getWorldPosition()).xyz;
}

/** @public-api */
vec3 getWorldViewVector() {
    return shading_view;
}

bool isPerspectiveProjection() {
    return frameUniforms.clipFromViewMatrix[2].w != 0.0;
}

/** @public-api */
vec3 getWorldNormalVector() {
    return shading_normal;
}

/** @public-api */
vec3 getWorldGeometricNormalVector() {
    return shading_geometricNormal;
}

/** @public-api */
vec3 getWorldReflectedVector() {
    return shading_reflected;
}

/** @public-api */
float getNdotV() {
    return shading_NoV;
}

highp vec3 getNormalizedPhysicalViewportCoord() {
    // make sure to handle our reversed-z
    return vec3(shading_normalizedViewportCoord, gl_FragCoord.z);
}

/**
 * Returns the normalized [0, 1] logical viewport coordinates with the origin at the viewport's
 * bottom-left. Z coordinate is in the [1, 0] range (reversed).
 *
 * @public-api
 */
highp vec3 getNormalizedViewportCoord() {
    // make sure to handle our reversed-z
    highp vec2 scale = frameUniforms.logicalViewportScale;
    highp vec2 offset = frameUniforms.logicalViewportOffset;
    highp vec2 logicalUv = shading_normalizedViewportCoord * scale + offset;
    return vec3(logicalUv, gl_FragCoord.z);
}


#if defined(MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY)
bool isDoubleSided() {
    return materialParams._doubleSided;
}
#endif


#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                     
#endif
// Decide if we can skip lighting when dot(n, l) <= 0.0
#if defined(SHADING_MODEL_CLOTH)
#if !defined(MATERIAL_HAS_SUBSURFACE_COLOR)
    #define MATERIAL_CAN_SKIP_LIGHTING
#endif
#elif defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_CUSTOM_SURFACE_SHADING)
    // Cannot skip lighting
#else
    #define MATERIAL_CAN_SKIP_LIGHTING
#endif

struct MaterialInputs {
    vec4  baseColor;
#if !defined(SHADING_MODEL_UNLIT)
#if !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    float roughness;
#endif
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    float metallic;
    float reflectance;
#endif
    float ambientOcclusion;
#endif
    vec4  emissive;

#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE) && !defined(SHADING_MODEL_UNLIT)
    vec3 sheenColor;
    float sheenRoughness;
#endif

    float clearCoat;
    float clearCoatRoughness;

    float anisotropy;
    vec3  anisotropyDirection;

#if defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_REFRACTION)
    float thickness;
#endif
#if defined(SHADING_MODEL_SUBSURFACE)
    float subsurfacePower;
    vec3  subsurfaceColor;
#endif

#if defined(SHADING_MODEL_CLOTH)
    vec3  sheenColor;
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
    vec3  subsurfaceColor;
#endif
#endif

#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    vec3  specularColor;
    float glossiness;
#endif

#if defined(MATERIAL_HAS_NORMAL)
    vec3  normal;
#endif
#if defined(MATERIAL_HAS_BENT_NORMAL)
    vec3  bentNormal;
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT) && defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    vec3  clearCoatNormal;
#endif

#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE) && !defined(SHADING_MODEL_UNLIT)
#if defined(MATERIAL_HAS_REFRACTION)
#if defined(MATERIAL_HAS_ABSORPTION)
    vec3 absorption;
#endif
#if defined(MATERIAL_HAS_TRANSMISSION)
    float transmission;
#endif
#if defined(MATERIAL_HAS_IOR)
    float ior;
#endif
#if defined(MATERIAL_HAS_MICRO_THICKNESS) && (REFRACTION_TYPE == REFRACTION_TYPE_THIN)
    float microThickness;
#endif
#elif !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
#if defined(MATERIAL_HAS_IOR)
    float ior;
#endif
#endif
#endif

#if defined(MATERIAL_HAS_SPECULAR_FACTOR)
    float specularFactor;
#endif

#if defined(MATERIAL_HAS_SPECULAR_COLOR_FACTOR)
    vec3 specularColorFactor;
#endif

};

void initMaterial(out MaterialInputs material) {
    material.baseColor = vec4(1.0);
#if !defined(SHADING_MODEL_UNLIT)
#if !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    material.roughness = 1.0;
#endif
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    material.metallic = 0.0;
    material.reflectance = 0.5;
#endif
    material.ambientOcclusion = 1.0;
#endif
    material.emissive = vec4(vec3(0.0), 1.0);

#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE) && !defined(SHADING_MODEL_UNLIT)
#if defined(MATERIAL_HAS_SHEEN_COLOR)
    material.sheenColor = vec3(0.0);
    material.sheenRoughness = 0.0;
#endif
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT)
    material.clearCoat = 1.0;
    material.clearCoatRoughness = 0.0;
#endif

#if defined(MATERIAL_HAS_ANISOTROPY)
    material.anisotropy = 0.0;
    material.anisotropyDirection = vec3(1.0, 0.0, 0.0);
#endif

#if defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_REFRACTION)
    material.thickness = 0.5;
#endif
#if defined(SHADING_MODEL_SUBSURFACE)
    material.subsurfacePower = 12.234;
    material.subsurfaceColor = vec3(1.0);
#endif

#if defined(SHADING_MODEL_CLOTH)
    material.sheenColor = sqrt(material.baseColor.rgb);
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
    material.subsurfaceColor = vec3(0.0);
#endif
#endif

#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    material.glossiness = 0.0;
    material.specularColor = vec3(0.0);
#endif

#if defined(MATERIAL_HAS_NORMAL)
    material.normal = vec3(0.0, 0.0, 1.0);
#endif
#if defined(MATERIAL_HAS_BENT_NORMAL)
    material.bentNormal = vec3(0.0, 0.0, 1.0);
#endif
#if defined(MATERIAL_HAS_CLEAR_COAT) && defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    material.clearCoatNormal = vec3(0.0, 0.0, 1.0);
#endif


#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE) && !defined(SHADING_MODEL_UNLIT)
#if defined(MATERIAL_HAS_REFRACTION)
#if defined(MATERIAL_HAS_ABSORPTION)
    material.absorption = vec3(0.0);
#endif
#if defined(MATERIAL_HAS_TRANSMISSION)
    material.transmission = 1.0;
#endif
#if defined(MATERIAL_HAS_IOR)
    material.ior = 1.5;
#endif
#if defined(MATERIAL_HAS_MICRO_THICKNESS) && (REFRACTION_TYPE == REFRACTION_TYPE_THIN)
    material.microThickness = 0.0;
#endif
#elif !defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
#if defined(MATERIAL_HAS_IOR)
    material.ior = 1.5;
#endif
#endif
#endif

#if defined(MATERIAL_HAS_SPECULAR_FACTOR)
    material.specularFactor = 1.0;
#endif

#if defined(MATERIAL_HAS_SPECULAR_COLOR_FACTOR)
    material.specularColorFactor = vec3(1.0);
#endif

}

#if defined(MATERIAL_HAS_CUSTOM_SURFACE_SHADING)
/** @public-api */
struct LightData {
    vec4  colorIntensity;
    vec3  l;
    float NdotL;
    vec3  worldPosition;
    float attenuation;
    float visibility;
};

/** @public-api */
struct ShadingData {
    vec3  diffuseColor;
    float perceptualRoughness;
    vec3  f0;
    float roughness;
};
#endif
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                        
#endif
//------------------------------------------------------------------------------
// Material evaluation
//------------------------------------------------------------------------------

/**
 * Computes global shading parameters used to apply lighting, such as the view
 * vector in world space, the tangent frame at the shading point, etc.
 */
void computeShadingParams() {
#if defined(HAS_ATTRIBUTE_TANGENTS)
    highp vec3 n = vertex_worldNormal;
#if defined(MATERIAL_NEEDS_TBN)
    highp vec3 t = vertex_worldTangent.xyz;
    highp vec3 b = cross(n, t) * sign(vertex_worldTangent.w);
#endif

#if defined(MATERIAL_HAS_DOUBLE_SIDED_CAPABILITY)
    if (isDoubleSided()) {
        n = gl_FrontFacing ? n : -n;
#if defined(MATERIAL_NEEDS_TBN)
        t = gl_FrontFacing ? t : -t;
        b = gl_FrontFacing ? b : -b;
#endif
    }
#endif

    shading_geometricNormal = normalize(n);

#if defined(MATERIAL_NEEDS_TBN)
    // We use unnormalized post-interpolation values, assuming mikktspace tangents
    shading_tangentToWorld = mat3(t, b, n);
#endif
#endif

    shading_position = vertex_worldPosition.xyz;

    // With perspective camera, the view vector is cast from the fragment pos to the eye position,
    // With ortho camera, however, the view vector is the same for all fragments:
    highp vec3 sv = isPerspectiveProjection() ?
        (frameUniforms.worldFromViewMatrix[3].xyz - shading_position) :
         frameUniforms.worldFromViewMatrix[2].xyz; // ortho camera backward dir
    shading_view = normalize(sv);

    // we do this so we avoid doing (matrix multiply), but we burn 4 varyings:
    //    p = clipFromWorldMatrix * shading_position;
    //    shading_normalizedViewportCoord = p.xy * 0.5 / p.w + 0.5
    shading_normalizedViewportCoord = vertex_position.xy * (0.5 / vertex_position.w) + 0.5;
}

/**
 * Computes global shading parameters that the material might need to access
 * before lighting: N dot V, the reflected vector and the shading normal (before
 * applying the normal map). These parameters can be useful to material authors
 * to compute other material properties.
 *
 * This function must be invoked by the user's material code (guaranteed by
 * the material compiler) after setting a value for MaterialInputs.normal.
 */
void prepareMaterial(const MaterialInputs material) {
#if defined(HAS_ATTRIBUTE_TANGENTS)
#if defined(MATERIAL_HAS_NORMAL)
    shading_normal = normalize(shading_tangentToWorld * material.normal);
#else
    shading_normal = getWorldGeometricNormalVector();
#endif
    shading_NoV = clampNoV(dot(shading_normal, shading_view));
    shading_reflected = reflect(-shading_view, shading_normal);

#if defined(MATERIAL_HAS_BENT_NORMAL)
    shading_bentNormal = normalize(shading_tangentToWorld * material.bentNormal);
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT)
#if defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    shading_clearCoatNormal = normalize(shading_tangentToWorld * material.clearCoatNormal);
#else
    shading_clearCoatNormal = getWorldGeometricNormalVector();
#endif
#endif
#endif
}
#line 1
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0    
#endif
void material(inout MaterialInputs material) {

        prepareMaterial(material);
        material.baseColor = materialParams.baseColorFactor;
    
                material.roughness = materialParams.roughnessFactor;
                material.metallic = materialParams.metallicFactor;
                material.emissive = vec4(materialParams.emissiveStrength *
                    materialParams.emissiveFactor.rgb, 0.0);
            highp float2 metallicRoughnessUV = getUV0();

                    vec4 mr = texture(materialParams_metallicRoughnessMap, metallicRoughnessUV);
                    material.roughness *= mr.g;
                    material.metallic *= mr.b;
                highp float2 aoUV = getUV0();

                float occlusion = texture(materialParams_occlusionMap, aoUV).r;
                material.ambientOcclusion = 1.0 + materialParams.aoStrength * (occlusion - 1.0);
            
                material.specularFactor = materialParams.specularStrength;
                material.specularColorFactor = materialParams.specularColorFactor;
            }
#line 1445
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                     
#endif
struct Light {
    vec4 colorIntensity;  // rgb, pre-exposed intensity
    vec3 l;
    float attenuation;
    highp vec3 worldPosition;
    float NoL;
    highp vec3 direction;
    float zLight;
    bool castsShadows;
    bool contactShadows;
    uint type;
    int shadowIndex;
    int channels;
};

struct PixelParams {
    vec3  diffuseColor;
    float perceptualRoughness;
    float perceptualRoughnessUnclamped;
    vec3  f0;
#if defined(MATERIAL_HAS_SPECULAR_COLOR_FACTOR) || defined(MATERIAL_HAS_SPECULAR_FACTOR)
    float f90;
    float specular;
    vec3  specularColor;
#endif
    float roughness;
    vec3  dfg;
    vec3  energyCompensation;

#if defined(MATERIAL_HAS_CLEAR_COAT)
    float clearCoat;
    float clearCoatPerceptualRoughness;
    float clearCoatRoughness;
#endif

#if defined(MATERIAL_HAS_SHEEN_COLOR)
    vec3  sheenColor;
#if !defined(SHADING_MODEL_CLOTH)
    float sheenRoughness;
    float sheenPerceptualRoughness;
    float sheenScaling;
    float sheenDFG;
#endif
#endif

#if defined(MATERIAL_HAS_ANISOTROPY)
    vec3  anisotropicT;
    vec3  anisotropicB;
    float anisotropy;
#endif

#if defined(SHADING_MODEL_SUBSURFACE) || defined(MATERIAL_HAS_REFRACTION)
    float thickness;
#endif
#if defined(SHADING_MODEL_SUBSURFACE)
    vec3  subsurfaceColor;
    float subsurfacePower;
#endif

#if defined(SHADING_MODEL_CLOTH) && defined(MATERIAL_HAS_SUBSURFACE_COLOR)
    vec3  subsurfaceColor;
#endif

#if defined(MATERIAL_HAS_REFRACTION)
    float etaRI;
    float etaIR;
    float transmission;
    float uThickness;
    vec3  absorption;
#endif
};

float computeMicroShadowing(float NoL, float visibility) {
    // Chan 2018, "Material Advances in Call of Duty: WWII"
    float aperture = inversesqrt(1.0 - min(visibility, 0.9999));
    float microShadow = saturate(NoL * aperture);
    return microShadow * microShadow;
}


/**
 * Returns the reflected vector at the current shading point. The reflected vector
 * return by this function might be different from shading_reflected:
 * - For anisotropic material, we bend the reflection vector to simulate
 *   anisotropic indirect lighting
 * - The reflected vector may be modified to point towards the dominant specular
 *   direction to match reference renderings when the roughness increases
 */

vec3 getReflectedVector(const PixelParams pixel, const vec3 v, const vec3 n) {
#if defined(MATERIAL_HAS_ANISOTROPY)
    vec3  anisotropyDirection = pixel.anisotropy >= 0.0 ? pixel.anisotropicB : pixel.anisotropicT;
    vec3  anisotropicTangent  = cross(anisotropyDirection, v);
    vec3  anisotropicNormal   = cross(anisotropicTangent, anisotropyDirection);
    float bendFactor          = abs(pixel.anisotropy) * saturate(5.0 * pixel.perceptualRoughness);
    vec3  bentNormal          = normalize(mix(n, anisotropicNormal, bendFactor));

    vec3 r = reflect(-v, bentNormal);
#else
    vec3 r = reflect(-v, n);
#endif
    return r;
}

void getAnisotropyPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(MATERIAL_HAS_ANISOTROPY)
    vec3 direction = material.anisotropyDirection;
    pixel.anisotropy = material.anisotropy;
    pixel.anisotropicT = normalize(shading_tangentToWorld * direction);
    pixel.anisotropicB = normalize(cross(getWorldGeometricNormalVector(), pixel.anisotropicT));
#endif
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0          
#endif
//------------------------------------------------------------------------------
// BRDF configuration
//------------------------------------------------------------------------------

// Diffuse BRDFs
#define DIFFUSE_LAMBERT             0
#define DIFFUSE_BURLEY              1

// Specular BRDF
// Normal distribution functions
#define SPECULAR_D_GGX              0

// Anisotropic NDFs
#define SPECULAR_D_GGX_ANISOTROPIC  0

// Cloth NDFs
#define SPECULAR_D_CHARLIE          0

// Visibility functions
#define SPECULAR_V_SMITH_GGX        0
#define SPECULAR_V_SMITH_GGX_FAST   1
#define SPECULAR_V_GGX_ANISOTROPIC  2
#define SPECULAR_V_KELEMEN          3
#define SPECULAR_V_NEUBELT          4

// Fresnel functions
#define SPECULAR_F_SCHLICK          0

#define BRDF_DIFFUSE                DIFFUSE_LAMBERT

#if FILAMENT_QUALITY < FILAMENT_QUALITY_HIGH
#define BRDF_SPECULAR_D             SPECULAR_D_GGX
#define BRDF_SPECULAR_V             SPECULAR_V_SMITH_GGX_FAST
#define BRDF_SPECULAR_F             SPECULAR_F_SCHLICK
#else
#define BRDF_SPECULAR_D             SPECULAR_D_GGX
#define BRDF_SPECULAR_V             SPECULAR_V_SMITH_GGX
#define BRDF_SPECULAR_F             SPECULAR_F_SCHLICK
#endif

#define BRDF_CLEAR_COAT_D           SPECULAR_D_GGX
#define BRDF_CLEAR_COAT_V           SPECULAR_V_KELEMEN

#define BRDF_ANISOTROPIC_D          SPECULAR_D_GGX_ANISOTROPIC
#define BRDF_ANISOTROPIC_V          SPECULAR_V_GGX_ANISOTROPIC

#define BRDF_CLOTH_D                SPECULAR_D_CHARLIE
#define BRDF_CLOTH_V                SPECULAR_V_NEUBELT

//------------------------------------------------------------------------------
// Specular BRDF implementations
//------------------------------------------------------------------------------

float D_GGX(float roughness, float NoH, const vec3 h) {
    // Walter et al. 2007, "Microfacet Models for Refraction through Rough Surfaces"

    // In mediump, there are two problems computing 1.0 - NoH^2
    // 1) 1.0 - NoH^2 suffers floating point cancellation when NoH^2 is close to 1 (highlights)
    // 2) NoH doesn't have enough precision around 1.0
    // Both problem can be fixed by computing 1-NoH^2 in highp and providing NoH in highp as well

    // However, we can do better using Lagrange's identity:
    //      ||a x b||^2 = ||a||^2 ||b||^2 - (a . b)^2
    // since N and H are unit vectors: ||N x H||^2 = 1.0 - NoH^2
    // This computes 1.0 - NoH^2 directly (which is close to zero in the highlights and has
    // enough precision).
    // Overall this yields better performance, keeping all computations in mediump
#if defined(TARGET_MOBILE)
    vec3 NxH = cross(shading_normal, h);
    float oneMinusNoHSquared = dot(NxH, NxH);
#else
    float oneMinusNoHSquared = 1.0 - NoH * NoH;
#endif

    float a = NoH * roughness;
    float k = roughness / (oneMinusNoHSquared + a * a);
    float d = k * k * (1.0 / PI);
    return saturateMediump(d);
}

float D_GGX_Anisotropic(float at, float ab, float ToH, float BoH, float NoH) {
    // Burley 2012, "Physically-Based Shading at Disney"

    // The values at and ab are perceptualRoughness^2, a2 is therefore perceptualRoughness^4
    // The dot product below computes perceptualRoughness^8. We cannot fit in fp16 without clamping
    // the roughness to too high values so we perform the dot product and the division in fp32
    float a2 = at * ab;
    highp vec3 d = vec3(ab * ToH, at * BoH, a2 * NoH);
    highp float d2 = dot(d, d);
    float b2 = a2 / d2;
    return a2 * b2 * b2 * (1.0 / PI);
}

float D_Charlie(float roughness, float NoH) {
    // Estevez and Kulla 2017, "Production Friendly Microfacet Sheen BRDF"
    float invAlpha  = 1.0 / roughness;
    float cos2h = NoH * NoH;
    float sin2h = max(1.0 - cos2h, 0.0078125); // 2^(-14/2), so sin2h^2 > 0 in fp16
    return (2.0 + invAlpha) * pow(sin2h, invAlpha * 0.5) / (2.0 * PI);
}

float V_SmithGGXCorrelated(float roughness, float NoV, float NoL) {
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    float a2 = roughness * roughness;
    // TODO: lambdaV can be pre-computed for all the lights, it should be moved out of this function
    float lambdaV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
    float lambdaL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
    float v = 0.5 / (lambdaV + lambdaL);
    // a2=0 => v = 1 / 4*NoL*NoV   => min=1/4, max=+inf
    // a2=1 => v = 1 / 2*(NoL+NoV) => min=1/4, max=+inf
    // clamp to the maximum value representable in mediump
    return saturateMediump(v);
}

float V_SmithGGXCorrelated_Fast(float roughness, float NoV, float NoL) {
    // Hammon 2017, "PBR Diffuse Lighting for GGX+Smith Microsurfaces"
    float v = 0.5 / mix(2.0 * NoL * NoV, NoL + NoV, roughness);
    return saturateMediump(v);
}

float V_SmithGGXCorrelated_Anisotropic(float at, float ab, float ToV, float BoV,
        float ToL, float BoL, float NoV, float NoL) {
    // Heitz 2014, "Understanding the Masking-Shadowing Function in Microfacet-Based BRDFs"
    // TODO: lambdaV can be pre-computed for all the lights, it should be moved out of this function
    float lambdaV = NoL * length(vec3(at * ToV, ab * BoV, NoV));
    float lambdaL = NoV * length(vec3(at * ToL, ab * BoL, NoL));
    float v = 0.5 / (lambdaV + lambdaL);
    return saturateMediump(v);
}

float V_Kelemen(float LoH) {
    // Kelemen 2001, "A Microfacet Based Coupled Specular-Matte BRDF Model with Importance Sampling"
    return saturateMediump(0.25 / (LoH * LoH));
}

float V_Neubelt(float NoV, float NoL) {
    // Neubelt and Pettineo 2013, "Crafting a Next-gen Material Pipeline for The Order: 1886"
    return saturateMediump(1.0 / (4.0 * (NoL + NoV - NoL * NoV)));
}

vec3 F_Schlick(const vec3 f0, float f90, float VoH) {
    // Schlick 1994, "An Inexpensive BRDF Model for Physically-Based Rendering"
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

vec3 F_Schlick(const vec3 f0, float VoH) {
    float f = pow(1.0 - VoH, 5.0);
    return f + f0 * (1.0 - f);
}

float F_Schlick(float f0, float f90, float VoH) {
    return f0 + (f90 - f0) * pow5(1.0 - VoH);
}

//------------------------------------------------------------------------------
// Specular BRDF dispatch
//------------------------------------------------------------------------------

float distribution(float roughness, float NoH, const vec3 h) {
#if BRDF_SPECULAR_D == SPECULAR_D_GGX
    return D_GGX(roughness, NoH, h);
#endif
}

float visibility(float roughness, float NoV, float NoL) {
#if BRDF_SPECULAR_V == SPECULAR_V_SMITH_GGX
    return V_SmithGGXCorrelated(roughness, NoV, NoL);
#elif BRDF_SPECULAR_V == SPECULAR_V_SMITH_GGX_FAST
    return V_SmithGGXCorrelated_Fast(roughness, NoV, NoL);
#endif
}

vec3 fresnel(const vec3 f0, float LoH) {
#if BRDF_SPECULAR_F == SPECULAR_F_SCHLICK
#if FILAMENT_QUALITY == FILAMENT_QUALITY_LOW
    return F_Schlick(f0, LoH); // f90 = 1.0
#else
    float f90 = saturate(dot(f0, vec3(50.0 * 0.33)));
    return F_Schlick(f0, f90, LoH);
#endif
#endif
}

vec3 fresnel(const vec3 f0, const float f90, float LoH) {
#if BRDF_SPECULAR_F == SPECULAR_F_SCHLICK
#if FILAMENT_QUALITY == FILAMENT_QUALITY_LOW
    return F_Schlick(f0, LoH); // f90 = 1.0
#else
    return F_Schlick(f0, f90, LoH);
#endif
#endif
}

float distributionAnisotropic(float at, float ab, float ToH, float BoH, float NoH) {
#if BRDF_ANISOTROPIC_D == SPECULAR_D_GGX_ANISOTROPIC
    return D_GGX_Anisotropic(at, ab, ToH, BoH, NoH);
#endif
}

float visibilityAnisotropic(float roughness, float at, float ab,
        float ToV, float BoV, float ToL, float BoL, float NoV, float NoL) {
#if BRDF_ANISOTROPIC_V == SPECULAR_V_SMITH_GGX
    return V_SmithGGXCorrelated(roughness, NoV, NoL);
#elif BRDF_ANISOTROPIC_V == SPECULAR_V_GGX_ANISOTROPIC
    return V_SmithGGXCorrelated_Anisotropic(at, ab, ToV, BoV, ToL, BoL, NoV, NoL);
#endif
}

float distributionClearCoat(float roughness, float NoH, const vec3 h) {
#if BRDF_CLEAR_COAT_D == SPECULAR_D_GGX
    return D_GGX(roughness, NoH, h);
#endif
}

float visibilityClearCoat(float LoH) {
#if BRDF_CLEAR_COAT_V == SPECULAR_V_KELEMEN
    return V_Kelemen(LoH);
#endif
}

float distributionCloth(float roughness, float NoH) {
#if BRDF_CLOTH_D == SPECULAR_D_CHARLIE
    return D_Charlie(roughness, NoH);
#endif
}

float visibilityCloth(float NoV, float NoL) {
#if BRDF_CLOTH_V == SPECULAR_V_NEUBELT
    return V_Neubelt(NoV, NoL);
#endif
}

//------------------------------------------------------------------------------
// Diffuse BRDF implementations
//------------------------------------------------------------------------------

float Fd_Lambert() {
    return 1.0 / PI;
}

float Fd_Burley(float roughness, float NoV, float NoL, float LoH) {
    // Burley 2012, "Physically-Based Shading at Disney"
    float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
    float lightScatter = F_Schlick(1.0, f90, NoL);
    float viewScatter  = F_Schlick(1.0, f90, NoV);
    return lightScatter * viewScatter * (1.0 / PI);
}

// Energy conserving wrap diffuse term, does *not* include the divide by pi
float Fd_Wrap(float NoL, float w) {
    return saturate((NoL + w) / sq(1.0 + w));
}

//------------------------------------------------------------------------------
// Diffuse BRDF dispatch
//------------------------------------------------------------------------------

float diffuse(float roughness, float NoV, float NoL, float LoH) {
#if BRDF_DIFFUSE == DIFFUSE_LAMBERT
    return Fd_Lambert();
#elif BRDF_DIFFUSE == DIFFUSE_BURLEY
    return Fd_Burley(roughness, NoV, NoL, LoH);
#endif
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                            
#endif
#if defined(MATERIAL_HAS_SHEEN_COLOR)
vec3 sheenLobe(const PixelParams pixel, float NoV, float NoL, float NoH) {
    float D = distributionCloth(pixel.sheenRoughness, NoH);
    float V = visibilityCloth(NoV, NoL);

    return (D * V) * pixel.sheenColor;
}
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT)
float clearCoatLobe(const PixelParams pixel, const vec3 h, float NoH, float LoH, out float Fcc) {
#if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // If the material has a normal map, we want to use the geometric normal
    // instead to avoid applying the normal map details to the clear coat layer
    float clearCoatNoH = saturate(dot(shading_clearCoatNormal, h));
#else
    float clearCoatNoH = NoH;
#endif

    // clear coat specular lobe
    float D = distributionClearCoat(pixel.clearCoatRoughness, clearCoatNoH, h);
    float V = visibilityClearCoat(LoH);
    float F = F_Schlick(0.04, 1.0, LoH) * pixel.clearCoat; // fix IOR to 1.5

    Fcc = F;
    return D * V * F;
}
#endif

#if defined(MATERIAL_HAS_ANISOTROPY)
vec3 anisotropicLobe(const PixelParams pixel, const Light light, const vec3 h,
        float NoV, float NoL, float NoH, float LoH) {

    vec3 l = light.l;
    vec3 t = pixel.anisotropicT;
    vec3 b = pixel.anisotropicB;
    vec3 v = shading_view;

    float ToV = dot(t, v);
    float BoV = dot(b, v);
    float ToL = dot(t, l);
    float BoL = dot(b, l);
    float ToH = dot(t, h);
    float BoH = dot(b, h);

    // Anisotropic parameters: at and ab are the roughness along the tangent and bitangent
    // to simplify materials, we derive them from a single roughness parameter
    // Kulla 2017, "Revisiting Physically Based Shading at Imageworks"
    float at = max(pixel.roughness * (1.0 + pixel.anisotropy), MIN_ROUGHNESS);
    float ab = max(pixel.roughness * (1.0 - pixel.anisotropy), MIN_ROUGHNESS);

    // specular anisotropic BRDF
    float D = distributionAnisotropic(at, ab, ToH, BoH, NoH);
    float V = visibilityAnisotropic(pixel.roughness, at, ab, ToV, BoV, ToL, BoL, NoV, NoL);
    vec3  F = fresnel(pixel.f0, LoH);

    return (D * V) * F;
}
#endif

vec3 isotropicLobe(const PixelParams pixel, const Light light, const vec3 h,
        float NoV, float NoL, float NoH, float LoH) {

    float D = distribution(pixel.roughness, NoH, h);
    float V = visibility(pixel.roughness, NoV, NoL);
#if defined(MATERIAL_HAS_SPECULAR_COLOR_FACTOR) || defined(MATERIAL_HAS_SPECULAR_FACTOR)
    vec3  F = fresnel(pixel.f0, pixel.f90, LoH);
#else
    vec3  F = fresnel(pixel.f0, LoH);
#endif
    return (D * V) * F;
}

vec3 specularLobe(const PixelParams pixel, const Light light, const vec3 h,
        float NoV, float NoL, float NoH, float LoH) {
#if defined(MATERIAL_HAS_ANISOTROPY)
    return anisotropicLobe(pixel, light, h, NoV, NoL, NoH, LoH);
#else
    return isotropicLobe(pixel, light, h, NoV, NoL, NoH, LoH);
#endif
}

vec3 diffuseLobe(const PixelParams pixel, float NoV, float NoL, float LoH) {
    return pixel.diffuseColor * diffuse(pixel.roughness, NoV, NoL, LoH);
}

/**
 * Evaluates lit materials with the standard shading model. This model comprises
 * of 2 BRDFs: an optional clear coat BRDF, and a regular surface BRDF.
 *
 * Surface BRDF
 * The surface BRDF uses a diffuse lobe and a specular lobe to render both
 * dielectrics and conductors. The specular lobe is based on the Cook-Torrance
 * micro-facet model (see brdf.fs for more details). In addition, the specular
 * can be either isotropic or anisotropic.
 *
 * Clear coat BRDF
 * The clear coat BRDF simulates a transparent, absorbing dielectric layer on
 * top of the surface. Its IOR is set to 1.5 (polyutherane) to simplify
 * our computations. This BRDF only contains a specular lobe and while based
 * on the Cook-Torrance microfacet model, it uses cheaper terms than the surface
 * BRDF's specular lobe (see brdf.fs).
 */
vec3 surfaceShading(const PixelParams pixel, const Light light, float occlusion) {
    vec3 h = normalize(shading_view + light.l);

    float NoV = shading_NoV;
    float NoL = saturate(light.NoL);
    float NoH = saturate(dot(shading_normal, h));
    float LoH = saturate(dot(light.l, h));

    vec3 Fr = specularLobe(pixel, light, h, NoV, NoL, NoH, LoH);
    vec3 Fd = diffuseLobe(pixel, NoV, NoL, LoH);
#if defined(MATERIAL_HAS_REFRACTION)
    Fd *= (1.0 - pixel.transmission);
#endif

    // TODO: attenuate the diffuse lobe to avoid energy gain

    // The energy compensation term is used to counteract the darkening effect
    // at high roughness
    vec3 color = Fd + Fr * pixel.energyCompensation;

#if defined(MATERIAL_HAS_SHEEN_COLOR)
    color *= pixel.sheenScaling;
    color += sheenLobe(pixel, NoV, NoL, NoH);
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT)
    float Fcc;
    float clearCoat = clearCoatLobe(pixel, h, NoH, LoH, Fcc);
    float attenuation = 1.0 - Fcc;

#if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    color *= attenuation * NoL;

    // If the material has a normal map, we want to use the geometric normal
    // instead to avoid applying the normal map details to the clear coat layer
    float clearCoatNoL = saturate(dot(shading_clearCoatNormal, light.l));
    color += clearCoat * clearCoatNoL;

    // Early exit to avoid the extra multiplication by NoL
    return (color * light.colorIntensity.rgb) *
            (light.colorIntensity.w * light.attenuation * occlusion);
#else
    color *= attenuation;
    color += clearCoat;
#endif
#endif
    return (color * light.colorIntensity.rgb) *
            (light.colorIntensity.w * light.attenuation * NoL * occlusion);
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                       
#endif
//------------------------------------------------------------------------------
// Ambient occlusion configuration
//------------------------------------------------------------------------------

// Diffuse BRDFs
#define SPECULAR_AO_OFF             0
#define SPECULAR_AO_SIMPLE          1
#define SPECULAR_AO_BENT_NORMALS    2

//------------------------------------------------------------------------------
// Ambient occlusion helpers
//------------------------------------------------------------------------------

float unpack(vec2 depth) {
    // this is equivalent to (x8 * 256 + y8) / 65535, which gives a value between 0 and 1
    return (depth.x * (256.0 / 257.0) + depth.y * (1.0 / 257.0));
}

struct SSAOInterpolationCache {
    highp vec4 weights;
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED) || defined(MATERIAL_HAS_REFLECTIONS)
    highp vec2 uv;
#endif
};

float evaluateSSAO(inout SSAOInterpolationCache cache) {
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED)

    if (frameUniforms.aoSamplingQualityAndEdgeDistance < 0.0) {
        // SSAO is disabled
        return 1.0;
    }

    // Upscale the SSAO buffer in real-time, in high quality mode we use a custom bilinear
    // filter. This adds about 2.0ms @ 250MHz on Pixel 4.

    if (frameUniforms.aoSamplingQualityAndEdgeDistance > 0.0) {
        highp vec2 size = vec2(textureSize(sampler0_ssao, 0));

        // Read four AO samples and their depths values
#if defined(FILAMENT_HAS_FEATURE_TEXTURE_GATHER)
        vec4 ao = textureGather(sampler0_ssao, vec3(cache.uv, 0.0), 0);
        vec4 dg = textureGather(sampler0_ssao, vec3(cache.uv, 0.0), 1);
        vec4 db = textureGather(sampler0_ssao, vec3(cache.uv, 0.0), 2);
#else
        vec3 s01 = textureLodOffset(sampler0_ssao, vec3(cache.uv, 0.0), 0.0, ivec2(0, 1)).rgb;
        vec3 s11 = textureLodOffset(sampler0_ssao, vec3(cache.uv, 0.0), 0.0, ivec2(1, 1)).rgb;
        vec3 s10 = textureLodOffset(sampler0_ssao, vec3(cache.uv, 0.0), 0.0, ivec2(1, 0)).rgb;
        vec3 s00 = textureLodOffset(sampler0_ssao, vec3(cache.uv, 0.0), 0.0, ivec2(0, 0)).rgb;
        vec4 ao = vec4(s01.r, s11.r, s10.r, s00.r);
        vec4 dg = vec4(s01.g, s11.g, s10.g, s00.g);
        vec4 db = vec4(s01.b, s11.b, s10.b, s00.b);
#endif
        // bilateral weights
        vec4 depths;
        depths.x = unpack(vec2(dg.x, db.x));
        depths.y = unpack(vec2(dg.y, db.y));
        depths.z = unpack(vec2(dg.z, db.z));
        depths.w = unpack(vec2(dg.w, db.w));
        depths *= -frameUniforms.cameraFar;

        // bilinear weights
        vec2 f = fract(cache.uv * size - 0.5);
        vec4 b;
        b.x = (1.0 - f.x) * f.y;
        b.y = f.x * f.y;
        b.z = f.x * (1.0 - f.y);
        b.w = (1.0 - f.x) * (1.0 - f.y);

        highp mat4 m = getViewFromWorldMatrix();
        highp float d = dot(vec3(m[0].z, m[1].z, m[2].z), shading_position) + m[3].z;
        highp vec4 w = (vec4(d) - depths) * frameUniforms.aoSamplingQualityAndEdgeDistance;
        w = max(vec4(MEDIUMP_FLT_MIN), 1.0 - w * w) * b;
        cache.weights = w / (w.x + w.y + w.z + w.w);
        return dot(ao, cache.weights);
    } else {
        return textureLod(sampler0_ssao, vec3(cache.uv, 0.0), 0.0).r;
    }
#else
    // SSAO is not applied when blending is enabled
    return 1.0;
#endif
}

float SpecularAO_Lagarde(float NoV, float visibility, float roughness) {
    // Lagarde and de Rousiers 2014, "Moving Frostbite to PBR"
    return saturate(pow(NoV + visibility, exp2(-16.0 * roughness - 1.0)) - 1.0 + visibility);
}

float sphericalCapsIntersection(float cosCap1, float cosCap2, float cosDistance) {
    // Oat and Sander 2007, "Ambient Aperture Lighting"
    // Approximation mentioned by Jimenez et al. 2016
    float r1 = acosFastPositive(cosCap1);
    float r2 = acosFastPositive(cosCap2);
    float d  = acosFast(cosDistance);

    // We work with cosine angles, replace the original paper's use of
    // cos(min(r1, r2)) with max(cosCap1, cosCap2)
    // We also remove a multiplication by 2 * PI to simplify the computation
    // since we divide by 2 * PI in computeBentSpecularAO()

    if (min(r1, r2) <= max(r1, r2) - d) {
        return 1.0 - max(cosCap1, cosCap2);
    } else if (r1 + r2 <= d) {
        return 0.0;
    }

    float delta = abs(r1 - r2);
    float x = 1.0 - saturate((d - delta) / max(r1 + r2 - delta, 1e-4));
    // simplified smoothstep()
    float area = sq(x) * (-2.0 * x + 3.0);
    return area * (1.0 - max(cosCap1, cosCap2));
}

// This function could (should?) be implemented as a 3D LUT instead, but we need to save samplers
float SpecularAO_Cones(vec3 bentNormal, float visibility, float roughness) {
    // Jimenez et al. 2016, "Practical Realtime Strategies for Accurate Indirect Occlusion"

    // aperture from ambient occlusion
    float cosAv = sqrt(1.0 - visibility);
    // aperture from roughness, log(10) / log(2) = 3.321928
    float cosAs = exp2(-3.321928 * sq(roughness));
    // angle betwen bent normal and reflection direction
    float cosB  = dot(bentNormal, shading_reflected);

    // Remove the 2 * PI term from the denominator, it cancels out the same term from
    // sphericalCapsIntersection()
    float ao = sphericalCapsIntersection(cosAv, cosAs, cosB) / (1.0 - cosAs);
    // Smoothly kill specular AO when entering the perceptual roughness range [0.1..0.3]
    // Without this, specular AO can remove all reflections, which looks bad on metals
    return mix(1.0, ao, smoothstep(0.01, 0.09, roughness));
}

/**
 * Computes a specular occlusion term from the ambient occlusion term.
 */
vec3 unpackBentNormal(vec3 bn) {
    // this must match src/materials/ssao/ssaoUtils.fs
    return bn * 2.0 - 1.0;
}

float specularAO(float NoV, float visibility, float roughness, const in SSAOInterpolationCache cache) {
    float specularAO = 1.0;

// SSAO is not applied when blending is enabled
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED)

#if SPECULAR_AMBIENT_OCCLUSION == SPECULAR_AO_SIMPLE
    // TODO: Should we even bother computing this when screen space bent normals are enabled?
    specularAO = SpecularAO_Lagarde(NoV, visibility, roughness);
#elif SPECULAR_AMBIENT_OCCLUSION == SPECULAR_AO_BENT_NORMALS
#   if defined(MATERIAL_HAS_BENT_NORMAL)
        specularAO = SpecularAO_Cones(shading_bentNormal, visibility, roughness);
#   else
        specularAO = SpecularAO_Cones(shading_normal, visibility, roughness);
#   endif
#endif

    if (frameUniforms.aoBentNormals > 0.0) {
        vec3 bn;
        if (frameUniforms.aoSamplingQualityAndEdgeDistance > 0.0) {
#if defined(FILAMENT_HAS_FEATURE_TEXTURE_GATHER)
            vec4 bnr = textureGather(sampler0_ssao, vec3(cache.uv, 1.0), 0);
            vec4 bng = textureGather(sampler0_ssao, vec3(cache.uv, 1.0), 1);
            vec4 bnb = textureGather(sampler0_ssao, vec3(cache.uv, 1.0), 2);
#else
            vec3 s01 = textureLodOffset(sampler0_ssao, vec3(cache.uv, 1.0), 0.0, ivec2(0, 1)).rgb;
            vec3 s11 = textureLodOffset(sampler0_ssao, vec3(cache.uv, 1.0), 0.0, ivec2(1, 1)).rgb;
            vec3 s10 = textureLodOffset(sampler0_ssao, vec3(cache.uv, 1.0), 0.0, ivec2(1, 0)).rgb;
            vec3 s00 = textureLodOffset(sampler0_ssao, vec3(cache.uv, 1.0), 0.0, ivec2(0, 0)).rgb;
            vec4 bnr = vec4(s01.r, s11.r, s10.r, s00.r);
            vec4 bng = vec4(s01.g, s11.g, s10.g, s00.g);
            vec4 bnb = vec4(s01.b, s11.b, s10.b, s00.b);
#endif
            bn.r = dot(bnr, cache.weights);
            bn.g = dot(bng, cache.weights);
            bn.b = dot(bnb, cache.weights);
        } else {
            bn = textureLod(sampler0_ssao, vec3(cache.uv, 1.0), 0.0).xyz;
        }

        bn = unpackBentNormal(bn);
        bn = normalize(bn);

        float ssSpecularAO = SpecularAO_Cones(bn, visibility, roughness);
        // Combine the specular AO from the texture with screen space specular AO
        specularAO = min(specularAO, ssSpecularAO);

        // For now we don't use the screen space AO bent normal for the diffuse because the
        // AO bent normal is currently a face normal.
    }
#endif

    return specularAO;
}

#if MULTI_BOUNCE_AMBIENT_OCCLUSION == 1
/**
 * Returns a color ambient occlusion based on a pre-computed visibility term.
 * The albedo term is meant to be the diffuse color or f0 for the diffuse and
 * specular terms respectively.
 */
vec3 gtaoMultiBounce(float visibility, const vec3 albedo) {
    // Jimenez et al. 2016, "Practical Realtime Strategies for Accurate Indirect Occlusion"
    vec3 a =  2.0404 * albedo - 0.3324;
    vec3 b = -4.7951 * albedo + 0.6417;
    vec3 c =  2.7552 * albedo + 0.6903;

    return max(vec3(visibility), ((visibility * a + b) * visibility + c) * visibility);
}
#endif

void multiBounceAO(float visibility, const vec3 albedo, inout vec3 color) {
#if MULTI_BOUNCE_AMBIENT_OCCLUSION == 1
    color *= gtaoMultiBounce(visibility, albedo);
#endif
}

void multiBounceSpecularAO(float visibility, const vec3 albedo, inout vec3 color) {
#if MULTI_BOUNCE_AMBIENT_OCCLUSION == 1 && SPECULAR_AMBIENT_OCCLUSION != SPECULAR_AO_OFF
    color *= gtaoMultiBounce(visibility, albedo);
#endif
}

float singleBounceAO(float visibility) {
#if MULTI_BOUNCE_AMBIENT_OCCLUSION == 1
    return 1.0;
#else
    return visibility;
#endif
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                    
#endif
//------------------------------------------------------------------------------
// Image based lighting configuration
//------------------------------------------------------------------------------

// IBL integration algorithm
#define IBL_INTEGRATION_PREFILTERED_CUBEMAP         0
#define IBL_INTEGRATION_IMPORTANCE_SAMPLING         1

#define IBL_INTEGRATION                             IBL_INTEGRATION_PREFILTERED_CUBEMAP

#define IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT   64

//------------------------------------------------------------------------------
// IBL utilities
//------------------------------------------------------------------------------

vec3 decodeDataForIBL(const vec4 data) {
    return data.rgb;
}

//------------------------------------------------------------------------------
// IBL prefiltered DFG term implementations
//------------------------------------------------------------------------------

vec3 PrefilteredDFG_LUT(float lod, float NoV) {
    // coord = sqrt(linear_roughness), which is the mapping used by cmgen.
    return textureLod(sampler0_iblDFG, vec2(NoV, lod), 0.0).rgb;
}

//------------------------------------------------------------------------------
// IBL environment BRDF dispatch
//------------------------------------------------------------------------------

vec3 prefilteredDFG(float perceptualRoughness, float NoV) {
    // PrefilteredDFG_LUT() takes a LOD, which is sqrt(roughness) = perceptualRoughness
    return PrefilteredDFG_LUT(perceptualRoughness, NoV);
}

//------------------------------------------------------------------------------
// IBL irradiance implementations
//------------------------------------------------------------------------------

vec3 Irradiance_SphericalHarmonics(const vec3 n) {
    vec3 sphericalHarmonics = frameUniforms.iblSH[0];

    if (CONFIG_SH_BANDS_COUNT >= 2) {
        sphericalHarmonics +=
                  frameUniforms.iblSH[1] * (n.y)
                + frameUniforms.iblSH[2] * (n.z)
                + frameUniforms.iblSH[3] * (n.x);
    }

    if (CONFIG_SH_BANDS_COUNT >= 3) {
        sphericalHarmonics +=
                  frameUniforms.iblSH[4] * (n.y * n.x)
                + frameUniforms.iblSH[5] * (n.y * n.z)
                + frameUniforms.iblSH[6] * (3.0 * n.z * n.z - 1.0)
                + frameUniforms.iblSH[7] * (n.z * n.x)
                + frameUniforms.iblSH[8] * (n.x * n.x - n.y * n.y);
    }

    return max(sphericalHarmonics, 0.0);
}

vec3 Irradiance_RoughnessOne(const vec3 n) {
    // note: lod used is always integer, hopefully the hardware skips tri-linear filtering
    return decodeDataForIBL(textureLod(sampler0_iblSpecular, n, frameUniforms.iblRoughnessOneLevel));
}

//------------------------------------------------------------------------------
// IBL irradiance dispatch
//------------------------------------------------------------------------------

vec3 diffuseIrradiance(const vec3 n) {
    // On Metal devices with certain chipsets, this light_iblSpecular texture sample must be pulled
    // outside the frameUniforms.iblSH check. This is to avoid a Metal pipeline compilation error
    // with the message: "Could not statically determine the target of a texture".
    // The reason for this is unknown, and is possibly a bug that exhibits only on these devices.
    vec3 irradianceRoughnessOne;
    if (CONFIG_STATIC_TEXTURE_TARGET_WORKAROUND) {
        irradianceRoughnessOne = Irradiance_RoughnessOne(n);
    }

    if (frameUniforms.iblSH[0].x == 65504.0) {
#if FILAMENT_QUALITY < FILAMENT_QUALITY_HIGH
        if (CONFIG_STATIC_TEXTURE_TARGET_WORKAROUND) {
            return irradianceRoughnessOne;
        } else {
            return Irradiance_RoughnessOne(n);
        }
#else
        ivec2 s = textureSize(sampler0_iblSpecular, int(frameUniforms.iblRoughnessOneLevel));
        float du = 1.0 / float(s.x);
        float dv = 1.0 / float(s.y);
        vec3 m0 = normalize(cross(n, vec3(0.0, 1.0, 0.0)));
        vec3 m1 = cross(m0, n);
        vec3 m0du = m0 * du;
        vec3 m1dv = m1 * dv;
        vec3 c;
        c  = Irradiance_RoughnessOne(n - m0du - m1dv);
        c += Irradiance_RoughnessOne(n + m0du - m1dv);
        c += Irradiance_RoughnessOne(n + m0du + m1dv);
        c += Irradiance_RoughnessOne(n - m0du + m1dv);
        return c * 0.25;
#endif
    } else {
        return Irradiance_SphericalHarmonics(n);
    }
}

//------------------------------------------------------------------------------
// IBL specular
//------------------------------------------------------------------------------

float perceptualRoughnessToLod(float perceptualRoughness) {
    // The mapping below is a quadratic fit for log2(perceptualRoughness)+iblRoughnessOneLevel when
    // iblRoughnessOneLevel is 4. We found empirically that this mapping works very well for
    // a 256 cubemap with 5 levels used. But also scales well for other iblRoughnessOneLevel values.
    return frameUniforms.iblRoughnessOneLevel * perceptualRoughness * (2.0 - perceptualRoughness);
}

vec3 prefilteredRadiance(const vec3 r, float perceptualRoughness) {
    float lod = perceptualRoughnessToLod(perceptualRoughness);
    return decodeDataForIBL(textureLod(sampler0_iblSpecular, r, lod));
}

vec3 prefilteredRadiance(const vec3 r, float roughness, float offset) {
    float lod = frameUniforms.iblRoughnessOneLevel * roughness;
    return decodeDataForIBL(textureLod(sampler0_iblSpecular, r, lod + offset));
}

vec3 getSpecularDominantDirection(const vec3 n, const vec3 r, float roughness) {
    return mix(r, n, roughness * roughness);
}

vec3 specularDFG(const PixelParams pixel) {
#if defined(SHADING_MODEL_CLOTH)
    return pixel.f0 * pixel.dfg.z;
#else
#if defined(MATERIAL_HAS_SPECULAR_COLOR_FACTOR) || defined(MATERIAL_HAS_SPECULAR_FACTOR)
    return mix(pixel.dfg.xxx, pixel.dfg.yyy, pixel.f0) * pixel.specular;
#else
    return mix(pixel.dfg.xxx, pixel.dfg.yyy, pixel.f0);
#endif
#endif
}

vec3 getReflectedVector(const PixelParams pixel, const vec3 n) {
#if defined(MATERIAL_HAS_ANISOTROPY)
    vec3 r = getReflectedVector(pixel, shading_view, n);
#else
    vec3 r = shading_reflected;
#endif
    return getSpecularDominantDirection(n, r, pixel.roughness);
}

//------------------------------------------------------------------------------
// Prefiltered importance sampling
//------------------------------------------------------------------------------

#if IBL_INTEGRATION == IBL_INTEGRATION_IMPORTANCE_SAMPLING
vec2 hammersley(uint index) {
    const uint numSamples = uint(IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT);
    const float invNumSamples = 1.0 / float(numSamples);
    const float tof = 0.5 / float(0x80000000U);
    uint bits = index;
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return vec2(float(index) * invNumSamples, float(bits) * tof);
}

vec3 importanceSamplingNdfDggx(vec2 u, float roughness) {
    // Importance sampling D_GGX
    float a2 = roughness * roughness;
    float phi = 2.0 * PI * u.x;
    float cosTheta2 = (1.0 - u.y) / (1.0 + (a2 - 1.0) * u.y);
    float cosTheta = sqrt(cosTheta2);
    float sinTheta = sqrt(1.0 - cosTheta2);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

vec3 hemisphereCosSample(vec2 u) {
    float phi = 2.0 * PI * u.x;
    float cosTheta2 = 1.0 - u.y;
    float cosTheta = sqrt(cosTheta2);
    float sinTheta = sqrt(1.0 - cosTheta2);
    return vec3(cos(phi) * sinTheta, sin(phi) * sinTheta, cosTheta);
}

vec3 importanceSamplingVNdfDggx(vec2 u, float roughness, vec3 v) {
    // See: "A Simpler and Exact Sampling Routine for the GGX Distribution of Visible Normals", Eric Heitz
    float alpha = roughness;

    // stretch view
    v = normalize(vec3(alpha * v.x, alpha * v.y, v.z));

    // orthonormal basis
    vec3 up = abs(v.z) < 0.9999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 t = normalize(cross(up, v));
    vec3 b = cross(t, v);

    // sample point with polar coordinates (r, phi)
    float a = 1.0 / (1.0 + v.z);
    float r = sqrt(u.x);
    float phi = (u.y < a) ? u.y / a * PI : PI + (u.y - a) / (1.0 - a) * PI;
    float p1 = r * cos(phi);
    float p2 = r * sin(phi) * ((u.y < a) ? 1.0 : v.z);

    // compute normal
    vec3 h = p1 * t + p2 * b + sqrt(max(0.0, 1.0 - p1*p1 - p2*p2)) * v;

    // unstretch
    h = normalize(vec3(alpha * h.x, alpha * h.y, max(0.0, h.z)));
    return h;
}

float prefilteredImportanceSampling(float ipdf, float omegaP) {
    // See: "Real-time Shading with Filtered Importance Sampling", Jaroslav Krivanek
    // Prefiltering doesn't work with anisotropy
    const float numSamples = float(IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT);
    const float invNumSamples = 1.0 / float(numSamples);
    const float K = 4.0;
    float omegaS = invNumSamples * ipdf;
    float mipLevel = log2(K * omegaS / omegaP) * 0.5;    // log4
    return mipLevel;
}

vec3 isEvaluateSpecularIBL(const PixelParams pixel, const vec3 n, const vec3 v, const float NoV) {
    const int numSamples = IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT;
    const float invNumSamples = 1.0 / float(numSamples);
    const vec3 up = vec3(0.0, 0.0, 1.0);

    // TODO: for a true anisotropic BRDF, we need a real tangent space
    // tangent space
    mat3 T;
    T[0] = normalize(cross(up, n));
    T[1] = cross(n, T[0]);
    T[2] = n;

    // Random rotation around N per pixel
    const vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
    float a = 2.0 * PI * fract(m.z * fract(dot(gl_FragCoord.xy, m.xy)));
    float c = cos(a);
    float s = sin(a);
    mat3 R;
    R[0] = vec3( c, s, 0);
    R[1] = vec3(-s, c, 0);
    R[2] = vec3( 0, 0, 1);
    T *= R;

    float roughness = pixel.roughness;
    float dim = float(textureSize(sampler0_iblSpecular, 0).x);
    float omegaP = (4.0 * PI) / (6.0 * dim * dim);

    vec3 indirectSpecular = vec3(0.0);
    for (int i = 0; i < numSamples; i++) {
        vec2 u = hammersley(i);
        vec3 h = T * importanceSamplingNdfDggx(u, roughness);

        // Since anisotropy doesn't work with prefiltering, we use the same "faux" anisotropy
        // we do when we use the prefiltered cubemap
        vec3 l = getReflectedVector(pixel, v, h);

        // Compute this sample's contribution to the brdf
        float NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            float NoH = dot(n, h);
            float LoH = saturate(dot(l, h));

            // PDF inverse (we must use D_GGX() here, which is used to generate samples)
            float ipdf = (4.0 * LoH) / (D_GGX(roughness, NoH, h) * NoH);
            float mipLevel = prefilteredImportanceSampling(ipdf, omegaP);
            vec3 L = decodeDataForIBL(textureLod(sampler0_iblSpecular, l, mipLevel));

            float D = distribution(roughness, NoH, h);
            float V = visibility(roughness, NoV, NoL);
            vec3 F = fresnel(pixel.f0, LoH);
            vec3 Fr = F * (D * V * NoL * ipdf * invNumSamples);

            indirectSpecular += (Fr * L);
        }
    }

    return indirectSpecular;
}

vec3 isEvaluateDiffuseIBL(const PixelParams pixel, vec3 n, vec3 v) {
    const int numSamples = IBL_INTEGRATION_IMPORTANCE_SAMPLING_COUNT;
    const float invNumSamples = 1.0 / float(numSamples);
    const vec3 up = vec3(0.0, 0.0, 1.0);

    // TODO: for a true anisotropic BRDF, we need a real tangent space
    // tangent space
    mat3 T;
    T[0] = normalize(cross(up, n));
    T[1] = cross(n, T[0]);
    T[2] = n;

    // Random rotation around N per pixel
    const vec3 m = vec3(0.06711056, 0.00583715, 52.9829189);
    float a = 2.0 * PI * fract(m.z * fract(dot(gl_FragCoord.xy, m.xy)));
    float c = cos(a);
    float s = sin(a);
    mat3 R;
    R[0] = vec3( c, s, 0);
    R[1] = vec3(-s, c, 0);
    R[2] = vec3( 0, 0, 1);
    T *= R;

    float dim = float(textureSize(sampler0_iblSpecular, 0).x);
    float omegaP = (4.0 * PI) / (6.0 * dim * dim);

    vec3 indirectDiffuse = vec3(0.0);
    for (int i = 0; i < numSamples; i++) {
        vec2 u = hammersley(i);
        vec3 h = T * hemisphereCosSample(u);

        // Since anisotropy doesn't work with prefiltering, we use the same "faux" anisotropy
        // we do when we use the prefiltered cubemap
        vec3 l = getReflectedVector(pixel, v, h);

        // Compute this sample's contribution to the brdf
        float NoL = saturate(dot(n, l));
        if (NoL > 0.0) {
            // PDF inverse (we must use D_GGX() here, which is used to generate samples)
            float ipdf = PI / NoL;
            // we have to bias the mipLevel (+1) to help with very strong highlights
            float mipLevel = prefilteredImportanceSampling(ipdf, omegaP) + 1.0;
            vec3 L = decodeDataForIBL(textureLod(sampler0_iblSpecular, l, mipLevel));
            indirectDiffuse += L;
        }
    }

    return indirectDiffuse * invNumSamples; // we bake 1/PI here, which cancels out
}

void isEvaluateClearCoatIBL(const PixelParams pixel, float specularAO, inout vec3 Fd, inout vec3 Fr) {
#if defined(MATERIAL_HAS_CLEAR_COAT)
#if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
    // We want to use the geometric normal for the clear coat layer
    float clearCoatNoV = clampNoV(dot(shading_clearCoatNormal, shading_view));
    vec3 clearCoatNormal = shading_clearCoatNormal;
#else
    float clearCoatNoV = shading_NoV;
    vec3 clearCoatNormal = shading_normal;
#endif
    // The clear coat layer assumes an IOR of 1.5 (4% reflectance)
    float Fc = F_Schlick(0.04, 1.0, clearCoatNoV) * pixel.clearCoat;
    float attenuation = 1.0 - Fc;
    Fd *= attenuation;
    Fr *= attenuation;

    PixelParams p;
    p.perceptualRoughness = pixel.clearCoatPerceptualRoughness;
    p.f0 = vec3(0.04);
    p.roughness = perceptualRoughnessToRoughness(p.perceptualRoughness);
#if defined(MATERIAL_HAS_ANISOTROPY)
    p.anisotropy = 0.0;
#endif

    vec3 clearCoatLobe = isEvaluateSpecularIBL(p, clearCoatNormal, shading_view, clearCoatNoV);
    Fr += clearCoatLobe * (specularAO * pixel.clearCoat);
#endif
}
#endif

//------------------------------------------------------------------------------
// IBL evaluation
//------------------------------------------------------------------------------

void evaluateClothIndirectDiffuseBRDF(const PixelParams pixel, inout float diffuse) {
#if defined(SHADING_MODEL_CLOTH)
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
    // Simulate subsurface scattering with a wrap diffuse term
    diffuse *= Fd_Wrap(shading_NoV, 0.5);
#endif
#endif
}

void evaluateSheenIBL(const PixelParams pixel, float diffuseAO,
        const in SSAOInterpolationCache cache, inout vec3 Fd, inout vec3 Fr) {
#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE)
#if defined(MATERIAL_HAS_SHEEN_COLOR)
    // Albedo scaling of the base layer before we layer sheen on top
    Fd *= pixel.sheenScaling;
    Fr *= pixel.sheenScaling;

    vec3 reflectance = pixel.sheenDFG * pixel.sheenColor;
    reflectance *= specularAO(shading_NoV, diffuseAO, pixel.sheenRoughness, cache);

    Fr += reflectance * prefilteredRadiance(shading_reflected, pixel.sheenPerceptualRoughness);
#endif
#endif
}

void evaluateClearCoatIBL(const PixelParams pixel, float diffuseAO,
        const in SSAOInterpolationCache cache, inout vec3 Fd, inout vec3 Fr) {
#if IBL_INTEGRATION == IBL_INTEGRATION_IMPORTANCE_SAMPLING
    float specularAO = specularAO(shading_NoV, diffuseAO, pixel.clearCoatRoughness, cache);
    isEvaluateClearCoatIBL(pixel, specularAO, Fd, Fr);
    return;
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT)
    if (pixel.clearCoat > 0.0) {
#if defined(MATERIAL_HAS_NORMAL) || defined(MATERIAL_HAS_CLEAR_COAT_NORMAL)
        // We want to use the geometric normal for the clear coat layer
        float clearCoatNoV = clampNoV(dot(shading_clearCoatNormal, shading_view));
        vec3 clearCoatR = reflect(-shading_view, shading_clearCoatNormal);
#else
        float clearCoatNoV = shading_NoV;
        vec3 clearCoatR = shading_reflected;
#endif
        // The clear coat layer assumes an IOR of 1.5 (4% reflectance)
        float Fc = F_Schlick(0.04, 1.0, clearCoatNoV) * pixel.clearCoat;
        float attenuation = 1.0 - Fc;
        Fd *= attenuation;
        Fr *= attenuation;

        // TODO: Should we apply specularAO to the attenuation as well?
        float specularAO = specularAO(clearCoatNoV, diffuseAO, pixel.clearCoatRoughness, cache);
        Fr += prefilteredRadiance(clearCoatR, pixel.clearCoatPerceptualRoughness) * (specularAO * Fc);
    }
#endif
}

void evaluateSubsurfaceIBL(const PixelParams pixel, const vec3 diffuseIrradiance,
        inout vec3 Fd, inout vec3 Fr) {
#if defined(SHADING_MODEL_SUBSURFACE)
    vec3 viewIndependent = diffuseIrradiance;
    vec3 viewDependent = prefilteredRadiance(-shading_view, pixel.roughness, 1.0 + pixel.thickness);
    float attenuation = (1.0 - pixel.thickness) / (2.0 * PI);
    Fd += pixel.subsurfaceColor * (viewIndependent + viewDependent) * attenuation;
#elif defined(SHADING_MODEL_CLOTH) && defined(MATERIAL_HAS_SUBSURFACE_COLOR)
    Fd *= saturate(pixel.subsurfaceColor + shading_NoV);
#endif
}

#if defined(MATERIAL_HAS_REFRACTION)

struct Refraction {
    vec3 position;
    vec3 direction;
    float d;
};

void refractionSolidSphere(const PixelParams pixel,
    const vec3 n, vec3 r, out Refraction ray) {
    r = refract(r, n, pixel.etaIR);
    float NoR = dot(n, r);
    float d = pixel.thickness * -NoR;
    ray.position = vec3(shading_position + r * d);
    ray.d = d;
    vec3 n1 = normalize(NoR * r - n * 0.5);
    ray.direction = refract(r, n1,  pixel.etaRI);
}

void refractionSolidBox(const PixelParams pixel,
    const vec3 n, vec3 r, out Refraction ray) {
    vec3 rr = refract(r, n, pixel.etaIR);
    float NoR = dot(n, rr);
    float d = pixel.thickness / max(-NoR, 0.001);
    ray.position = vec3(shading_position + rr * d);
    ray.direction = r;
    ray.d = d;
#if REFRACTION_MODE == REFRACTION_MODE_CUBEMAP
    // fudge direction vector, so we see the offset due to the thickness of the object
    float envDistance = 10.0; // this should come from a ubo
    ray.direction = normalize((ray.position - shading_position) + ray.direction * envDistance);
#endif
}

void refractionThinSphere(const PixelParams pixel,
    const vec3 n, vec3 r, out Refraction ray) {
    float d = 0.0;
#if defined(MATERIAL_HAS_MICRO_THICKNESS)
    // note: we need the refracted ray to calculate the distance traveled
    // we could use shading_NoV, but we would lose the dependency on ior.
    vec3 rr = refract(r, n, pixel.etaIR);
    float NoR = dot(n, rr);
    d = pixel.uThickness / max(-NoR, 0.001);
    ray.position = vec3(shading_position + rr * d);
#else
    ray.position = vec3(shading_position);
#endif
    ray.direction = r;
    ray.d = d;
}

vec3 evaluateRefraction(
    const PixelParams pixel,
    const vec3 n0, vec3 E) {

    Refraction ray;

#if REFRACTION_TYPE == REFRACTION_TYPE_SOLID
    refractionSolidSphere(pixel, n0, -shading_view, ray);
#elif REFRACTION_TYPE == REFRACTION_TYPE_THIN
    refractionThinSphere(pixel, n0, -shading_view, ray);
#else
#error invalid REFRACTION_TYPE
#endif

    // compute transmission T
#if defined(MATERIAL_HAS_ABSORPTION)
#if defined(MATERIAL_HAS_THICKNESS) || defined(MATERIAL_HAS_MICRO_THICKNESS)
    vec3 T = min(vec3(1.0), exp(-pixel.absorption * ray.d));
#else
    vec3 T = 1.0 - pixel.absorption;
#endif
#endif

    // Roughness remapping so that an IOR of 1.0 means no microfacet refraction and an IOR
    // of 1.5 has full microfacet refraction
    float perceptualRoughness = mix(pixel.perceptualRoughnessUnclamped, 0.0,
            saturate(pixel.etaIR * 3.0 - 2.0));
#if REFRACTION_TYPE == REFRACTION_TYPE_THIN
    // For thin surfaces, the light will bounce off at the second interface in the direction of
    // the reflection, effectively adding to the specular, but this process will repeat itself.
    // Each time the ray exits the surface on the front side after the first bounce,
    // it's multiplied by E^2, and we get: E + E(1-E)^2 + E^3(1-E)^2 + ...
    // This infinite series converges and is easy to simplify.
    // Note: we calculate these bounces only on a single component,
    // since it's a fairly subtle effect.
    E *= 1.0 + pixel.transmission * (1.0 - E.g) / (1.0 + E.g);
#endif

    /* sample the cubemap or screen-space */
#if REFRACTION_MODE == REFRACTION_MODE_CUBEMAP
    // when reading from the cubemap, we are not pre-exposed so we apply iblLuminance
    // which is not the case when we'll read from the screen-space buffer
    vec3 Ft = prefilteredRadiance(ray.direction, perceptualRoughness) * frameUniforms.iblLuminance;
#else
    vec3 Ft;

    // compute the point where the ray exits the medium, if needed
    vec4 p = vec4(getClipFromWorldMatrix() * vec4(ray.position, 1.0));
    p.xy = uvToRenderTargetUV(p.xy * (0.5 / p.w) + 0.5);

    // distance to camera plane
    const float invLog2sqrt5 = 0.8614;
    float lod = max(0.0, (2.0 * log2(perceptualRoughness) + frameUniforms.refractionLodOffset) * invLog2sqrt5);
    Ft = textureLod(sampler0_ssr, vec3(p.xy, 0.0), lod).rgb;
#endif

    // base color changes the amount of light passing through the boundary
    Ft *= pixel.diffuseColor;

    // fresnel from the first interface
    Ft *= 1.0 - E;

    // apply absorption
#if defined(MATERIAL_HAS_ABSORPTION)
    Ft *= T;
#endif

    return Ft;
}
#endif

void evaluateIBL(const MaterialInputs material, const PixelParams pixel, inout vec3 color) {
    // specular layer
    vec3 Fr = vec3(0.0);

    SSAOInterpolationCache interpolationCache;
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED) || defined(MATERIAL_HAS_REFLECTIONS)
    interpolationCache.uv = uvToRenderTargetUV(getNormalizedPhysicalViewportCoord().xy);
#endif

    // screen-space reflections
#if defined(MATERIAL_HAS_REFLECTIONS)
    vec4 ssrFr = vec4(0.0);
#if defined(BLEND_MODE_OPAQUE) || defined(BLEND_MODE_MASKED)
    // do the uniform based test first
    if (frameUniforms.ssrDistance > 0.0) {
        // There is no point doing SSR for very high roughness because we're limited by the fov
        // of the screen, in addition it doesn't really add much to the final image.
        // TODO: maybe make this a parameter
        const float maxPerceptualRoughness = sqrt(0.5);
        if (pixel.perceptualRoughness < maxPerceptualRoughness) {
            // distance to camera plane
            const float invLog2sqrt5 = 0.8614;
            float d = -mulMat4x4Float3(getViewFromWorldMatrix(), getWorldPosition()).z;
            float lod = max(0.0, (log2(pixel.roughness / d) + frameUniforms.refractionLodOffset) * invLog2sqrt5);
            ssrFr = textureLod(sampler0_ssr, vec3(interpolationCache.uv, 1.0), lod);
        }
    }
#else // BLEND_MODE_OPAQUE
    // TODO: for blended transparency, we have to ray-march here (limited to mirror reflections)
#endif
#else // MATERIAL_HAS_REFLECTIONS
    const vec4 ssrFr = vec4(0.0);
#endif

    // If screen-space reflections are turned on and have full contribution (ssr.a == 1.0), then we
    // skip sampling the IBL down below.

#if IBL_INTEGRATION == IBL_INTEGRATION_PREFILTERED_CUBEMAP
    vec3 E = specularDFG(pixel);
    if (ssrFr.a < 1.0) { // prevent reading the IBL if possible
        vec3 r = getReflectedVector(pixel, shading_normal);
        Fr = E * prefilteredRadiance(r, pixel.perceptualRoughness);
    }
#elif IBL_INTEGRATION == IBL_INTEGRATION_IMPORTANCE_SAMPLING
    vec3 E = vec3(0.0); // TODO: fix for importance sampling
    if (ssrFr.a < 1.0) { // prevent evaluating the IBL if possible
        Fr = isEvaluateSpecularIBL(pixel, shading_normal, shading_view, shading_NoV);
    }
#endif

    // Ambient occlusion
    float ssao = evaluateSSAO(interpolationCache);
    float diffuseAO = min(material.ambientOcclusion, ssao);
    float specularAO = specularAO(shading_NoV, diffuseAO, pixel.roughness, interpolationCache);

    vec3 specularSingleBounceAO = singleBounceAO(specularAO) * pixel.energyCompensation;
    Fr *= specularSingleBounceAO;
#if defined(MATERIAL_HAS_REFLECTIONS)
    ssrFr.rgb *= specularSingleBounceAO;
#endif

    // diffuse layer
    float diffuseBRDF = singleBounceAO(diffuseAO); // Fd_Lambert() is baked in the SH below
    evaluateClothIndirectDiffuseBRDF(pixel, diffuseBRDF);

#if defined(MATERIAL_HAS_BENT_NORMAL)
    vec3 diffuseNormal = shading_bentNormal;
#else
    vec3 diffuseNormal = shading_normal;
#endif

#if IBL_INTEGRATION == IBL_INTEGRATION_PREFILTERED_CUBEMAP
    vec3 diffuseIrradiance = diffuseIrradiance(diffuseNormal);
#elif IBL_INTEGRATION == IBL_INTEGRATION_IMPORTANCE_SAMPLING
    vec3 diffuseIrradiance = isEvaluateDiffuseIBL(pixel, diffuseNormal, shading_view);
#endif
    vec3 Fd = pixel.diffuseColor * diffuseIrradiance * (1.0 - E) * diffuseBRDF;

    // subsurface layer
    evaluateSubsurfaceIBL(pixel, diffuseIrradiance, Fd, Fr);

    // extra ambient occlusion term for the base and subsurface layers
    multiBounceAO(diffuseAO, pixel.diffuseColor, Fd);
    multiBounceSpecularAO(specularAO, pixel.f0, Fr);

    // sheen layer
    evaluateSheenIBL(pixel, diffuseAO, interpolationCache, Fd, Fr);

    // clear coat layer
    evaluateClearCoatIBL(pixel, diffuseAO, interpolationCache, Fd, Fr);

    Fr *= frameUniforms.iblLuminance;
    Fd *= frameUniforms.iblLuminance;

#if defined(MATERIAL_HAS_REFRACTION)
    vec3 Ft = evaluateRefraction(pixel, shading_normal, E);
    Ft *= pixel.transmission;
    Fd *= (1.0 - pixel.transmission);
#endif

#if defined(MATERIAL_HAS_REFLECTIONS)
    Fr = Fr * (1.0 - ssrFr.a) + (E * ssrFr.rgb);
#endif

    // Combine all terms
    // Note: iblLuminance is already premultiplied by the exposure
    color.rgb += Fr + Fd;
#if defined(MATERIAL_HAS_REFRACTION)
    color.rgb += Ft;
#endif
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                       
#endif
//------------------------------------------------------------------------------
// Directional light evaluation
//------------------------------------------------------------------------------

#if FILAMENT_QUALITY >= FILAMENT_QUALITY_HIGH
#define SUN_AS_AREA_LIGHT
#endif

vec3 sampleSunAreaLight(const vec3 lightDirection) {
#if defined(SUN_AS_AREA_LIGHT)
    if (frameUniforms.sun.w >= 0.0) {
        // simulate sun as disc area light
        float LoR = dot(lightDirection, shading_reflected);
        float d = frameUniforms.sun.x;
        highp vec3 s = shading_reflected - LoR * lightDirection;
        return LoR < d ?
                normalize(lightDirection * d + normalize(s) * frameUniforms.sun.y) : shading_reflected;
    }
#endif
    return lightDirection;
}

Light getDirectionalLight() {
    Light light;
    // note: lightColorIntensity.w is always premultiplied by the exposure
    light.colorIntensity = frameUniforms.lightColorIntensity;
    light.l = sampleSunAreaLight(frameUniforms.lightDirection);
    light.attenuation = 1.0;
    light.NoL = saturate(dot(shading_normal, light.l));
    light.channels = frameUniforms.lightChannels & 0xFF;
    return light;
}

void evaluateDirectionalLight(const MaterialInputs material,
        const PixelParams pixel, inout vec3 color) {

    Light light = getDirectionalLight();

    int channels = object_uniforms_flagsChannels & 0xFF;
    if ((light.channels & channels) == 0) {
        return;
    }

    float visibility = 1.0;


#if defined(MATERIAL_HAS_CUSTOM_SURFACE_SHADING)
    color.rgb += customSurfaceShading(material, pixel, light, visibility);
#else
    color.rgb += surfaceShading(pixel, light, visibility);
#endif
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0                 
#endif
//------------------------------------------------------------------------------
// Lighting
//------------------------------------------------------------------------------

#if defined(BLEND_MODE_MASKED)
float computeDiffuseAlpha(float a) {
    // If we reach this point in the code, we already know that the fragment is not discarded due
    // to the threshold factor. Therefore we can just output 1.0, which prevents a "punch through"
    // effect from occuring. We do this only for TRANSLUCENT views in order to prevent breakage
    // of ALPHA_TO_COVERAGE.
    return (frameUniforms.needsAlphaChannel == 1.0) ? 1.0 : a;
}
#else // not masked

float computeDiffuseAlpha(float a) {
#if defined(BLEND_MODE_TRANSPARENT) || defined(BLEND_MODE_FADE)
    return a;
#else
    return 1.0;
#endif
}
#endif

#if defined(GEOMETRIC_SPECULAR_AA)
float normalFiltering(float perceptualRoughness, const vec3 worldNormal) {
    // Kaplanyan 2016, "Stable specular highlights"
    // Tokuyoshi 2017, "Error Reduction and Simplification for Shading Anti-Aliasing"
    // Tokuyoshi and Kaplanyan 2019, "Improved Geometric Specular Antialiasing"

    // This implementation is meant for deferred rendering in the original paper but
    // we use it in forward rendering as well (as discussed in Tokuyoshi and Kaplanyan
    // 2019). The main reason is that the forward version requires an expensive transform
    // of the half vector by the tangent frame for every light. This is therefore an
    // approximation but it works well enough for our needs and provides an improvement
    // over our original implementation based on Vlachos 2015, "Advanced VR Rendering".

    vec3 du = dFdx(worldNormal);
    vec3 dv = dFdy(worldNormal);

    // specular AA factor to correct for resolution scaling (DSR and TAAx4)
    du *= frameUniforms.derivativesScale.x;
    dv *= frameUniforms.derivativesScale.y;

    float variance = materialParams._specularAntiAliasingVariance * (dot(du, du) + dot(dv, dv));
    float roughness = perceptualRoughnessToRoughness(perceptualRoughness);
    float kernelRoughness = min(2.0 * variance, materialParams._specularAntiAliasingThreshold);
    float squareRoughness = saturate(roughness * roughness + kernelRoughness);

    return roughnessToPerceptualRoughness(sqrt(squareRoughness));
}
#endif

void getCommonPixelParams(const MaterialInputs material, inout PixelParams pixel) {
    vec4 baseColor = material.baseColor;

#if defined(BLEND_MODE_FADE) && !defined(SHADING_MODEL_UNLIT)
    // Since we work in premultiplied alpha mode, we need to un-premultiply
    // in fade mode so we can apply alpha to both the specular and diffuse
    // components at the end
    unpremultiply(baseColor);
#endif

#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    // This is from KHR_materials_pbrSpecularGlossiness.
    vec3 specularColor = material.specularColor;
    float metallic = computeMetallicFromSpecularColor(specularColor);

    pixel.diffuseColor = computeDiffuseColor(baseColor, metallic);
    pixel.f0 = specularColor;
#elif !defined(SHADING_MODEL_CLOTH)
    pixel.diffuseColor = computeDiffuseColor(baseColor, material.metallic);
#if !defined(SHADING_MODEL_SUBSURFACE) && (!defined(MATERIAL_HAS_REFLECTANCE) && defined(MATERIAL_HAS_IOR))
    float reflectance = iorToF0(max(1.0, material.ior), 1.0);
#else
    // Assumes an interface from air to an IOR of 1.5 for dielectrics
    float reflectance = computeDielectricF0(material.reflectance);
#endif
#if !defined(MATERIAL_HAS_SPECULAR_FACTOR) && !defined(MATERIAL_HAS_SPECULAR_COLOR_FACTOR)
    pixel.f0 = computeF0(baseColor, material.metallic, reflectance);
#else
    vec3 dielectricSpecularF0 = vec3(0.0);
    float dielectricSpecularF90 = 0.0;
#if defined(MATERIAL_HAS_SPECULAR_COLOR_FACTOR)
    dielectricSpecularF0 = min(reflectance * material.specularColorFactor, vec3(1.0));
#endif
#if defined(MATERIAL_HAS_SPECULAR_FACTOR)
    dielectricSpecularF0 *= material.specularFactor;
    dielectricSpecularF90 = material.specularFactor;
#endif
    pixel.f0 = baseColor.rgb * material.metallic + dielectricSpecularF0 * (1.0 - material.metallic);
    pixel.f90 = material.metallic + dielectricSpecularF90 * (1.0 - material.metallic);
#endif
#else
    pixel.diffuseColor = baseColor.rgb;
    pixel.f0 = material.sheenColor;
#if defined(MATERIAL_HAS_SUBSURFACE_COLOR)
    pixel.subsurfaceColor = material.subsurfaceColor;
#endif
#endif

#if !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE)
#if defined(MATERIAL_HAS_REFRACTION)
    // Air's Index of refraction is 1.000277 at STP but everybody uses 1.0
    const float airIor = 1.0;
#if !defined(MATERIAL_HAS_IOR)
    // [common case] ior is not set in the material, deduce it from F0
    float materialor = f0ToIor(pixel.f0.g);
#else
    // if ior is set in the material, use it (can lead to unrealistic materials)
    float materialor = max(1.0, material.ior);
#endif
    pixel.etaIR = airIor / materialor;  // air -> material
    pixel.etaRI = materialor / airIor;  // material -> air
#if defined(MATERIAL_HAS_TRANSMISSION)
    pixel.transmission = saturate(material.transmission);
#else
    pixel.transmission = 1.0;
#endif
#if defined(MATERIAL_HAS_ABSORPTION)
#if defined(MATERIAL_HAS_THICKNESS) || defined(MATERIAL_HAS_MICRO_THICKNESS)
    pixel.absorption = max(vec3(0.0), material.absorption);
#else
    pixel.absorption = saturate(material.absorption);
#endif
#else
    pixel.absorption = vec3(0.0);
#endif
#if defined(MATERIAL_HAS_THICKNESS)
    pixel.thickness = max(0.0, material.thickness);
#endif
#if defined(MATERIAL_HAS_MICRO_THICKNESS) && (REFRACTION_TYPE == REFRACTION_TYPE_THIN)
    pixel.uThickness = max(0.0, material.microThickness);
#else
    pixel.uThickness = 0.0;
#endif
#endif
#endif
}

void getSpecularPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(MATERIAL_HAS_SPECULAR_FACTOR)
    pixel.specular = material.specularFactor;
#endif

#if defined(MATERIAL_HAS_SPECULAR_COLOR_FACTOR)
    pixel.specularColor = material.specularColorFactor;
#endif
}

void getSheenPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(MATERIAL_HAS_SHEEN_COLOR) && !defined(SHADING_MODEL_CLOTH) && !defined(SHADING_MODEL_SUBSURFACE)
    pixel.sheenColor = material.sheenColor;

    float sheenPerceptualRoughness = material.sheenRoughness;
    sheenPerceptualRoughness = clamp(sheenPerceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);

#if defined(GEOMETRIC_SPECULAR_AA)
    sheenPerceptualRoughness =
            normalFiltering(sheenPerceptualRoughness, getWorldGeometricNormalVector());
#endif

    pixel.sheenPerceptualRoughness = sheenPerceptualRoughness;
    pixel.sheenRoughness = perceptualRoughnessToRoughness(sheenPerceptualRoughness);
#endif
}

void getClearCoatPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(MATERIAL_HAS_CLEAR_COAT)
    pixel.clearCoat = material.clearCoat;

    // Clamp the clear coat roughness to avoid divisions by 0
    float clearCoatPerceptualRoughness = material.clearCoatRoughness;
    clearCoatPerceptualRoughness =
            clamp(clearCoatPerceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);

#if defined(GEOMETRIC_SPECULAR_AA)
    clearCoatPerceptualRoughness =
            normalFiltering(clearCoatPerceptualRoughness, getWorldGeometricNormalVector());
#endif

    pixel.clearCoatPerceptualRoughness = clearCoatPerceptualRoughness;
    pixel.clearCoatRoughness = perceptualRoughnessToRoughness(clearCoatPerceptualRoughness);

#if defined(CLEAR_COAT_IOR_CHANGE)
    // The base layer's f0 is computed assuming an interface from air to an IOR
    // of 1.5, but the clear coat layer forms an interface from IOR 1.5 to IOR
    // 1.5. We recompute f0 by first computing its IOR, then reconverting to f0
    // by using the correct interface
    pixel.f0 = mix(pixel.f0, f0ClearCoatToSurface(pixel.f0), pixel.clearCoat);
#endif
#endif
}

void getRoughnessPixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(SHADING_MODEL_SPECULAR_GLOSSINESS)
    float perceptualRoughness = computeRoughnessFromGlossiness(material.glossiness);
#else
    float perceptualRoughness = material.roughness;
#endif

    // This is used by the refraction code and must be saved before we apply specular AA
    pixel.perceptualRoughnessUnclamped = perceptualRoughness;

#if defined(GEOMETRIC_SPECULAR_AA)
    perceptualRoughness = normalFiltering(perceptualRoughness, getWorldGeometricNormalVector());
#endif

#if defined(MATERIAL_HAS_CLEAR_COAT) && defined(MATERIAL_HAS_CLEAR_COAT_ROUGHNESS)
    // This is a hack but it will do: the base layer must be at least as rough
    // as the clear coat layer to take into account possible diffusion by the
    // top layer
    float basePerceptualRoughness = max(perceptualRoughness, pixel.clearCoatPerceptualRoughness);
    perceptualRoughness = mix(perceptualRoughness, basePerceptualRoughness, pixel.clearCoat);
#endif

    // Clamp the roughness to a minimum value to avoid divisions by 0 during lighting
    pixel.perceptualRoughness = clamp(perceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS, 1.0);
    // Remaps the roughness to a perceptually linear roughness (roughness^2)
    pixel.roughness = perceptualRoughnessToRoughness(pixel.perceptualRoughness);
}

void getSubsurfacePixelParams(const MaterialInputs material, inout PixelParams pixel) {
#if defined(SHADING_MODEL_SUBSURFACE)
    pixel.subsurfacePower = material.subsurfacePower;
    pixel.subsurfaceColor = material.subsurfaceColor;
    pixel.thickness = saturate(material.thickness);
#endif
}

void getEnergyCompensationPixelParams(inout PixelParams pixel) {
    // Pre-filtered DFG term used for image-based lighting
    pixel.dfg = prefilteredDFG(pixel.perceptualRoughness, shading_NoV);

#if !defined(SHADING_MODEL_CLOTH)
    // Energy compensation for multiple scattering in a microfacet model
    // See "Multiple-Scattering Microfacet BSDFs with the Smith Model"
    pixel.energyCompensation = 1.0 + pixel.f0 * (1.0 / pixel.dfg.y - 1.0);
#else
    pixel.energyCompensation = vec3(1.0);
#endif

#if !defined(SHADING_MODEL_CLOTH)
#if defined(MATERIAL_HAS_SHEEN_COLOR)
    pixel.sheenDFG = prefilteredDFG(pixel.sheenPerceptualRoughness, shading_NoV).z;
    pixel.sheenScaling = 1.0 - max3(pixel.sheenColor) * pixel.sheenDFG;
#endif
#endif
}

/**
 * Computes all the parameters required to shade the current pixel/fragment.
 * These parameters are derived from the MaterialInputs structure computed
 * by the user's material code.
 *
 * This function is also responsible for discarding the fragment when alpha
 * testing fails.
 */
void getPixelParams(const MaterialInputs material, out PixelParams pixel) {
    getSpecularPixelParams(material, pixel);
    getCommonPixelParams(material, pixel);
    getSheenPixelParams(material, pixel);
    getClearCoatPixelParams(material, pixel);
    getRoughnessPixelParams(material, pixel);
    getSubsurfacePixelParams(material, pixel);
    getAnisotropyPixelParams(material, pixel);
    getEnergyCompensationPixelParams(pixel);
}

/**
 * This function evaluates all lights one by one:
 * - Image based lights (IBL)
 * - Directional lights
 * - Punctual lights
 *
 * Area lights are currently not supported.
 *
 * Returns a pre-exposed HDR RGBA color in linear space.
 */
vec4 evaluateLights(const MaterialInputs material) {
    PixelParams pixel;
    getPixelParams(material, pixel);

    // Ideally we would keep the diffuse and specular components separate
    // until the very end but it costs more ALUs on mobile. The gains are
    // currently not worth the extra operations
    vec3 color = vec3(0.0);

    // We always evaluate the IBL as not having one is going to be uncommon,
    // it also saves 1 shader variant
    evaluateIBL(material, pixel, color);

    //核心光照计算
    evaluateDirectionalLight(material, pixel, color);


#if defined(VARIANT_HAS_DYNAMIC_LIGHTING)
    evaluatePunctualLights(material, pixel, color);
#endif

#if defined(BLEND_MODE_FADE) && !defined(SHADING_MODEL_UNLIT)
    // In fade mode we un-premultiply baseColor early on, so we need to
    // premultiply again at the end (affects diffuse and specular lighting)
    color *= material.baseColor.a;
#endif

    return vec4(color, computeDiffuseAlpha(material.baseColor.a));
}

void addEmissive(const MaterialInputs material, inout vec4 color) {
#if defined(MATERIAL_HAS_EMISSIVE)
    highp vec4 emissive = material.emissive;
    highp float attenuation = mix(1.0, getExposure(), emissive.w);
#if defined(BLEND_MODE_TRANSPARENT) || defined(BLEND_MODE_FADE)
    attenuation *= color.a;
#endif
    color.rgb += emissive.rgb * attenuation;
#endif
}

/**
 * Evaluate lit materials. The actual shading model used to do so is defined
 * by the function surfaceShading() found in shading_model_*.fs.
 *
 * Returns a pre-exposed HDR RGBA color in linear space.
 */
vec4 evaluateMaterial(const MaterialInputs material) {
    vec4 color = evaluateLights(material);
    addEmissive(material, color);
    return color;
}
#if defined(GL_GOOGLE_cpp_style_line_directive)
#line 0          
#endif
#if __VERSION__ == 100
vec4 fragColor;
#else
layout(location = 0) out vec4 fragColor;
#endif


#if defined(BLEND_MODE_MASKED)
void applyAlphaMask(inout vec4 baseColor) {
    // Use derivatives to sharpen alpha tested edges, combined with alpha to
    // coverage to smooth the result
    baseColor.a = (baseColor.a - getMaskThreshold()) / max(fwidth(baseColor.a), 1e-3) + 0.5;
    if (baseColor.a <= getMaskThreshold()) {
        discard;
    }
}
#else
void applyAlphaMask(inout vec4 baseColor) {}
#endif

void main() {

#if defined(FILAMENT_HAS_FEATURE_INSTANCING)
    logical_instance_index = instance_index;
#endif

    initObjectUniforms();

    // See shading_parameters.fs
    // Computes global variables we need to evaluate material and lighting
    computeShadingParams();

    // Initialize the inputs to sensible default values, see material_inputs.fs
    MaterialInputs inputs;
    initMaterial(inputs);

    // Invoke user code
    material(inputs);

    applyAlphaMask(inputs.baseColor);

    fragColor = evaluateMaterial(inputs);



}

