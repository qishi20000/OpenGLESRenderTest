#include <jni.h>
#include <android/log.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <stdlib.h>  // 添加此头文件以使用malloc和free
#include <math.h>
#include <string.h>  // 添加此头文件以使用memcpy
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>

static const char* TAG = "OpenGLESTriangle";

// OBJ加载器结构体
struct Vertex {
    float x, y, z;
    float nx, ny, nz;
    float u, v;
};

struct OBJMesh {
    std::vector<Vertex> vertices;
    std::vector<unsigned short> indices;
    int vertexCount;
    int indexCount;
};

// Asset Manager
static AAssetManager* g_assetManager = nullptr;

// OBJ文件加载函数
static std::string readAssetFile(const char* filename) {
    if (!g_assetManager) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Asset manager not initialized");
        return "";
    }
    
    AAsset* asset = AAssetManager_open(g_assetManager, filename, AASSET_MODE_BUFFER);
    if (!asset) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to open asset: %s", filename);
        return "";
    }
    
    size_t length = AAsset_getLength(asset);
    char* buffer = (char*)malloc(length + 1);
    AAsset_read(asset, buffer, length);
    AAsset_close(asset);
    
    buffer[length] = '\0';
    std::string content(buffer);
    free(buffer);
    
    return content;
}

static OBJMesh loadOBJFile(const char* filename) {
    OBJMesh mesh;
    std::string content = readAssetFile(filename);
    
    if (content.empty()) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to read OBJ file: %s", filename);
        return mesh;
    }
    
    std::vector<float> positions, normals, texCoords;
    std::vector<std::string> faces;
    
    std::istringstream stream(content);
    std::string line;
    
    while (std::getline(stream, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream lineStream(line);
        std::string type;
        lineStream >> type;
        
        if (type == "v") {
            // 顶点位置
            float x, y, z;
            lineStream >> x >> y >> z;
            positions.push_back(x);
            positions.push_back(y);
            positions.push_back(z);
        }
        else if (type == "vn") {
            // 法线
            float nx, ny, nz;
            lineStream >> nx >> ny >> nz;
            normals.push_back(nx);
            normals.push_back(ny);
            normals.push_back(nz);
        }
        else if (type == "vt") {
            // 纹理坐标
            float u, v;
            lineStream >> u >> v;
            texCoords.push_back(u);
            texCoords.push_back(v);
        }
        else if (type == "f") {
            // 面
            faces.push_back(line);
        }
    }
    
    // 处理面数据
    for (const auto& face : faces) {
        std::istringstream faceStream(face);
        std::string token;
        faceStream >> token; // 跳过 "f"
        
        std::vector<int> vIndices, vtIndices, vnIndices;
        
        while (faceStream >> token) {
            std::istringstream tokenStream(token);
            std::string v, vt, vn;
            
            std::getline(tokenStream, v, '/');
            std::getline(tokenStream, vt, '/');
            std::getline(tokenStream, vn, '/');
            
            if (!v.empty()) vIndices.push_back(std::stoi(v) - 1);
            if (!vt.empty()) vtIndices.push_back(std::stoi(vt) - 1);
            if (!vn.empty()) vnIndices.push_back(std::stoi(vn) - 1);
        }
        
        // 创建三角形（假设是三角形面）
        if (vIndices.size() >= 3) {
            for (int i = 1; i < vIndices.size() - 1; i++) {
                // 添加三个顶点
                for (int j = 0; j < 3; j++) {
                    int idx = (j == 0) ? 0 : (j == 1) ? i : i + 1;
                    if (idx < vIndices.size()) {
                        Vertex vertex;
                        
                        // 位置
                        int posIdx = vIndices[idx] * 3;
                        vertex.x = positions[posIdx];
                        vertex.y = positions[posIdx + 1];
                        vertex.z = positions[posIdx + 2];
                        
                        // 法线
                        if (vnIndices.size() > idx) {
                            int normIdx = vnIndices[idx] * 3;
                            vertex.nx = normals[normIdx];
                            vertex.ny = normals[normIdx + 1];
                            vertex.nz = normals[normIdx + 2];
                        } else {
                            vertex.nx = 0; vertex.ny = 0; vertex.nz = 1;
                        }
                        
                        // 纹理坐标
                        if (vtIndices.size() > idx) {
                            int texIdx = vtIndices[idx] * 2;
                            vertex.u = texCoords[texIdx];
                            vertex.v = texCoords[texIdx + 1];
                        } else {
                            vertex.u = 0; vertex.v = 0;
                        }
                        
                        mesh.vertices.push_back(vertex);
                        mesh.indices.push_back(mesh.vertices.size() - 1);
                    }
                }
            }
        }
    }
    
    mesh.vertexCount = mesh.vertices.size();
    mesh.indexCount = mesh.indices.size();
    
    __android_log_print(ANDROID_LOG_INFO, TAG, "Loaded OBJ: %d vertices, %d indices", 
                       mesh.vertexCount, mesh.indexCount);
    
    return mesh;
}

// Shader sources
static const char* vertexShaderCode = R"(
        #version 300 es
        layout(location = 0) in vec3 aPosition;
        layout(location = 1) in vec3 aNormal;
        layout(location = 2) in vec2 aUV0;
        
        uniform mat4 uModelMatrix;
        uniform mat4 uViewMatrix;
        uniform mat4 uProjectionMatrix;
        
        out vec3 vWorldPosition;
        out vec3 vWorldNormal;
        out vec2 vUV0;
        out vec4 vPosition;
        
        void main() {
          vec4 worldPos = uModelMatrix * vec4(aPosition, 1.0);
          vWorldPosition = worldPos.xyz;
          vWorldNormal = mat3(uModelMatrix) * aNormal;
          vUV0 = aUV0;
          
          vec4 viewPos = uViewMatrix * worldPos;
          vPosition = uProjectionMatrix * viewPos;
          gl_Position = vPosition;
        }
        )";

static const char* fragmentShaderCode = R"(
        #version 300 es
        precision mediump float;
        
        in vec3 vWorldPosition;
        in vec3 vWorldNormal;
        in vec2 vUV0;
        in vec4 vPosition;
        
        // 光照参数
        uniform vec3 uCameraPosition;
        uniform vec3 uLightDirection;
        uniform vec4 uLightColorIntensity;
        
        // 材质参数（按照 Filament 标准）
        uniform vec4 uBaseColorFactor;
        uniform float uMetallicFactor;
        uniform float uRoughnessFactor;
        uniform float uNormalScale;
        uniform float uAOStrength;
        uniform vec3 uEmissiveFactor;
        uniform float uEmissiveStrength;
        uniform float uSpecularStrength;
        uniform vec3 uSpecularColorFactor;
        
        out vec4 fragColor;
        
        const float PI = 3.14159265359;
        const float MIN_PERCEPTUAL_ROUGHNESS = 0.089;
        const float MIN_ROUGHNESS = 0.007921;
        const float MIN_N_DOT_V = 1e-4;
        
        // Filament 工具函数
        float saturate(float x) {
          return clamp(x, 0.0, 1.0);
        }
        
        vec3 saturate(vec3 x) {
          return clamp(x, 0.0, 1.0);
        }
        
        float pow5(float x) {
          float x2 = x * x;
          return x2 * x2 * x;
        }
        
        // 简单的 PBR 函数
        vec3 computeDiffuseColor(const vec4 baseColor, float metallic) {
          return baseColor.rgb * (1.0 - metallic);
        }
        
        vec3 computeF0(const vec4 baseColor, float metallic, float reflectance) {
          return baseColor.rgb * metallic + (reflectance * (1.0 - metallic));
        }
        
        float computeDielectricF0(float reflectance) {
          return 0.16 * reflectance * reflectance;
        }
        
        // Filament 精确的 BRDF 函数实现
        float D_GGX(float roughness, float NoH, const vec3 h) {
          // Filament 的精确 GGX 实现，包括移动端优化
          float a = NoH * roughness;
          float k = roughness / (1.0 - NoH * NoH + a * a);
          float d = k * k * (1.0 / PI);
          return clamp(d, 0.0, 65504.0);
        }
        
        float V_SmithGGXCorrelated(float roughness, float NoV, float NoL) {
          // Filament 的 Smith GGX 相关实现 - Heitz 2014
          float a2 = roughness * roughness;
          float lambdaV = NoL * sqrt((NoV - a2 * NoV) * NoV + a2);
          float lambdaL = NoV * sqrt((NoL - a2 * NoL) * NoL + a2);
          float v = 0.5 / (lambdaV + lambdaL);
          return clamp(v, 0.0, 65504.0);
        }
        
        vec3 F_Schlick(const vec3 f0, float f90, float VoH) {
          // Filament 的 Schlick Fresnel 实现 - Schlick 1994
          return f0 + (f90 - f0) * pow5(1.0 - VoH);
        }
        
        vec3 F_Schlick(const vec3 f0, float VoH) {
          // Filament 的简化 Schlick Fresnel (f90 = 1.0)
          float f = pow5(1.0 - VoH);
          return f + f0 * (1.0 - f);
        }
        
        // Filament 的漫反射 BRDF
        float Fd_Lambert() {
          return 1.0 / PI;
        }
        
        float Fd_Burley(float roughness, float NoV, float NoL, float LoH) {
          // Filament 的 Burley 漫反射实现 - Disney 2012
          float f90 = 0.5 + 2.0 * roughness * LoH * LoH;
          float lightScatter = F_Schlick(vec3(1.0), f90, NoL).r;
          float viewScatter = F_Schlick(vec3(1.0), f90, NoV).r;
          return lightScatter * viewScatter * (1.0 / PI);
        }
        
        // Filament 的 DFG 预计算项（改进的近似）
        vec3 prefilteredDFG(float perceptualRoughness, float NoV) {
          // 改进的 DFG 近似，更接近 Filament 的 LUT
          float roughness = perceptualRoughness * perceptualRoughness;
          vec3 f0 = vec3(0.04);
          float f90 = 1.0;
          
          // 更准确的 DFG 近似
          float F = F_Schlick(f0, f90, NoV).r;
          float G = 1.0 / (4.0 * NoV + 0.1); // 简化的几何项
          float D = 1.0 / (roughness * roughness + 0.1); // 简化的分布项
          
          return vec3(F, G, D);
        }
        
        // IBL 漫反射辐照度（改进版本）
        vec3 diffuseIrradiance(vec3 normal) {
          // 改进的环境光照，模拟球谐函数的效果
          float NdotY = normal.y;
          float NdotX = normal.x;
          float NdotZ = normal.z;
          
          // 简化的球谐函数近似 - 降低环境光强度以获得更好的明暗对比
          vec3 color = vec3(0.08, 0.08, 0.08); // 基础环境色 - 降低强度
          color += vec3(0.05, 0.05, 0.05) * NdotY; // 垂直渐变
          color += vec3(0.02, 0.02, 0.02) * NdotX; // 水平渐变
          color += vec3(0.01, 0.01, 0.01) * NdotZ; // 深度渐变
          
          return color;
        }
        
        // IBL 镜面反射（改进版本）
        vec3 prefilteredRadiance(vec3 reflection, float perceptualRoughness) {
          // 改进的镜面反射，模拟预过滤环境贴图
          float roughness = perceptualRoughness * perceptualRoughness;
          float lod = roughness * 6.0; // 更准确的 LOD 计算
          
          // 基于反射方向的颜色变化 - 降低强度以获得更好的明暗对比
          vec3 baseColor = vec3(0.15, 0.15, 0.15); // 降低基础反射强度
          float fresnel = pow(1.0 - max(dot(reflection, vec3(0.0, 1.0, 0.0)), 0.0), 2.0);
          vec3 color = mix(baseColor, vec3(0.4, 0.4, 0.4), fresnel); // 降低高光强度
          
          // 基于粗糙度的衰减
          float attenuation = 1.0 - roughness;
          return color * attenuation;
        }
        
        // 计算镜面反射的 DFG 项（Filament 标准）
        vec3 specularDFG(vec3 f0, vec3 dfg) {
          return mix(dfg.xxx, dfg.yyy, f0);
        }
        
        void main() {
          // 获取材质参数（按照 Filament 标准）
          vec4 baseColor = uBaseColorFactor;
          float metallic = uMetallicFactor;
          float roughness = uRoughnessFactor;
          float ao = uAOStrength;
          vec3 emissive = uEmissiveFactor * uEmissiveStrength;
          float specularStrength = uSpecularStrength;
          vec3 specularColorFactor = uSpecularColorFactor;
          
          // Filament 的精确值限制
          const float MIN_PERCEPTUAL_ROUGHNESS = 0.089;
          const float MIN_ROUGHNESS = 0.007921;
          const float MIN_N_DOT_V = 1e-4;
          
          // 限制值范围（按照 Filament 标准）
          float perceptualRoughness = saturate(roughness);
          perceptualRoughness = max(perceptualRoughness, MIN_PERCEPTUAL_ROUGHNESS);
          roughness = max(perceptualRoughness * perceptualRoughness, MIN_ROUGHNESS);
          metallic = saturate(metallic);
          ao = saturate(ao);
          
          // 计算光照向量
          vec3 N = normalize(vWorldNormal);
          vec3 V = normalize(uCameraPosition - vWorldPosition);
          vec3 L = normalize(-uLightDirection);
          vec3 H = normalize(V + L);
          
          float NoV = max(dot(N, V), MIN_N_DOT_V);
          float NoL = saturate(dot(N, L));
          float NoH = saturate(dot(N, H));
          float VoH = saturate(dot(V, H));
          float LoH = saturate(dot(L, H));
          
          // 计算 F0（按照 Filament 标准，包括 specular 参数）
          float reflectance = computeDielectricF0(0.5);
          vec3 dielectricSpecularF0 = min(reflectance * specularColorFactor, vec3(1.0)) * specularStrength;
          vec3 f0 = baseColor.rgb * metallic + dielectricSpecularF0 * (1.0 - metallic);
          float f90 = 1.0; // Filament 标准
          
          // 镜面反射 BRDF（使用 Filament 的精确实现）
          float D = D_GGX(roughness, NoH, H);
          float V_brdf = V_SmithGGXCorrelated(roughness, NoV, NoL);
          vec3 F = F_Schlick(f0, f90, VoH);
          
          vec3 specular = (D * V_brdf) * F;
          
          // 漫反射 BRDF（使用 Filament 的 Burley 模型）
          vec3 diffuseColor = computeDiffuseColor(baseColor, metallic);
          float diffuse = Fd_Burley(roughness, NoV, NoL, LoH);
          vec3 diffuseBRDF = diffuseColor * diffuse;
          
          // 能量守恒（按照 Filament 标准）
          vec3 kS = F;
          vec3 kD = (vec3(1.0) - kS) * (1.0 - metallic);
          diffuseBRDF *= kD;
          
          // 计算 DFG 项（用于能量补偿和 IBL）
          vec3 dfg = prefilteredDFG(perceptualRoughness, NoV);
          
          // 能量补偿（Filament 的多散射补偿）
          vec3 energyCompensation = 1.0 + f0 * (1.0 / dfg.y - 1.0);
          specular *= energyCompensation;
          
          // 直接光照
          vec3 lightColor = uLightColorIntensity.rgb;
          float lightIntensity = uLightColorIntensity.a;
          vec3 radiance = lightColor * lightIntensity;
          
          vec3 directLighting = (diffuseBRDF + specular) * radiance * NoL;
          
          // IBL 计算（按照 Filament 标准）
          vec3 E = specularDFG(f0, dfg);
          
          // 漫反射 IBL（改进的能量守恒）
          vec3 diffuseIrradiance = diffuseIrradiance(N);
          vec3 Fd_IBL = diffuseColor * diffuseIrradiance * (1.0 - E) * (1.0 / PI);
          
          // 镜面反射 IBL（改进的实现）
          vec3 reflection = reflect(-V, N);
          vec3 Fr_IBL = E * prefilteredRadiance(reflection, perceptualRoughness);
          
          // 应用环境光遮蔽
          Fd_IBL *= ao;
          Fr_IBL *= ao;
          
          // 组合光照（按照 Filament 标准）
          vec3 color = Fd_IBL + Fr_IBL + directLighting;
          
          // 添加自发光（按照 Filament 标准）
          color += emissive;
          
          // 调试：确保颜色不为零
          if (length(color) < 0.01) {
            color = vec3(0.1, 0.1, 0.1); // 最小可见颜色
          }
          
          // 确保颜色在合理范围内
          color = saturate(color);
          
          fragColor = vec4(color, baseColor.a);
        }
        )";

static GLuint program = 0;
static GLint positionHandle = 0;
static GLint normalHandle = 0;
static GLint uv0Handle = 0;

// Matrix uniforms
static GLint uModelMatrixLocation = -1;
static GLint uViewMatrixLocation = -1;
static GLint uProjectionMatrixLocation = -1;
static GLint uNormalMatrixLocation = -1;

// Frame uniforms
static GLint uCameraPositionLocation = -1;
static GLint uLightDirectionLocation = -1;
static GLint uLightColorIntensityLocation = -1;
static GLint uExposureLocation = -1;
static GLint uIBLLuminanceLocation = -1;

// Material uniforms
static GLint uBaseColorFactorLocation = -1;
static GLint uMetallicFactorLocation = -1;
static GLint uRoughnessFactorLocation = -1;
static GLint uNormalScaleLocation = -1;
static GLint uAOStrengthLocation = -1;
static GLint uEmissiveFactorLocation = -1;
static GLint uEmissiveStrengthLocation = -1;
static GLint uSpecularStrengthLocation = -1;
static GLint uSpecularColorFactorLocation = -1;

// Texture uniforms
static GLint uMetallicRoughnessMapLocation = -1;
static GLint uOcclusionMapLocation = -1;

GLuint vbo = 0;
GLuint ibo = 0;
GLsizei objIndexCount = 0;
static OBJMesh g_objMesh;

// 旋转和缩放相关变量
static float rotationX = 0.0f;
static float rotationY = 0.0f;
static float scale = 1.0f;
static float lastTouchX = 0.0f;
static float lastTouchY = 0.0f;
static float lastDistance = 0.0f;
static bool isTouching = false;
static bool isTwoFinger = false;

// 辅助函数
static float calculateDistance(float x1, float y1, float x2, float y2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    return sqrtf(dx * dx + dy * dy);
}

// 矩阵计算函数
static void createIdentityMatrix(float* matrix) {
    memset(matrix, 0, 16 * sizeof(float));
    matrix[0] = matrix[5] = matrix[10] = matrix[15] = 1.0f;
}

static void createRotationMatrix(float* matrix, float angleX, float angleY) {
    createIdentityMatrix(matrix);
    
    // 绕X轴旋转矩阵
    float cosX = cosf(angleX);
    float sinX = sinf(angleX);
    
    // 绕Y轴旋转矩阵
    float cosY = cosf(angleY);
    float sinY = sinf(angleY);
    
    // 创建X轴旋转矩阵
    float rotX[16];
    createIdentityMatrix(rotX);
    rotX[5] = cosX;
    rotX[6] = -sinX;
    rotX[9] = sinX;
    rotX[10] = cosX;
    
    // 创建Y轴旋转矩阵
    float rotY[16];
    createIdentityMatrix(rotY);
    rotY[0] = cosY;
    rotY[2] = sinY;
    rotY[8] = -sinY;
    rotY[10] = cosY;
    
    // 矩阵乘法：result = rotY * rotX
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            matrix[i * 4 + j] = 0.0f;
            for (int k = 0; k < 4; k++) {
                matrix[i * 4 + j] += rotY[i * 4 + k] * rotX[k * 4 + j];
            }
        }
    }
}
//近大远小投影 
static void createPerspectiveMatrix(float* matrix, float fov, float aspect, float near, float far) {
    float f = 1.0f / tanf(fov * 0.5f * M_PI / 180.0f);
    float rangeInv = 1.0f / (near - far);
    
    createIdentityMatrix(matrix);
    matrix[0] = f / aspect;
    matrix[5] = f;
    matrix[10] = (near + far) * rangeInv;
    matrix[11] = -1.0f;
    matrix[14] = near * far * rangeInv * 2.0f;
    matrix[15] = 0.0f;
}
//正交投影
static void createOrthographicMatrix(float* matrix, float left, float right, float bottom, float top, float near, float far) {
    createIdentityMatrix(matrix);
    
    float rml = right - left;
    float tmb = top - bottom;
    float fmn = far - near;
    
    matrix[0] = 2.0f / rml;
    matrix[5] = 2.0f / tmb;
    matrix[10] = -2.0f / fmn;
    matrix[12] = -(right + left) / rml;
    matrix[13] = -(top + bottom) / tmb;
    matrix[14] = -(far + near) / fmn;
}

static void createViewMatrix(float* matrix, float eyeX, float eyeY, float eyeZ, 
                            float centerX, float centerY, float centerZ,
                            float upX, float upY, float upZ) {
    // 计算前向量
    float fx = centerX - eyeX;
    float fy = centerY - eyeY;
    float fz = centerZ - eyeZ;
    float length = sqrtf(fx * fx + fy * fy + fz * fz);
    fx /= length; fy /= length; fz /= length;
    
    // 计算右向量
    float rx = fy * upZ - fz * upY;
    float ry = fz * upX - fx * upZ;
    float rz = fx * upY - fy * upX;
    length = sqrtf(rx * rx + ry * ry + rz * rz);
    rx /= length; ry /= length; rz /= length;
    
    // 重新计算上向量
    float ux = ry * fz - rz * fy;
    float uy = rz * fx - rx * fz;
    float uz = rx * fy - ry * fx;
    
    createIdentityMatrix(matrix);
    matrix[0] = rx; matrix[1] = ry; matrix[2] = rz;
    matrix[4] = ux; matrix[5] = uy; matrix[6] = uz;
    matrix[8] = -fx; matrix[9] = -fy; matrix[10] = -fz;
    matrix[12] = -(rx * eyeX + ry * eyeY + rz * eyeZ);
    matrix[13] = -(ux * eyeX + uy * eyeY + uz * eyeZ);
    matrix[14] = -(-fx * eyeX - fy * eyeY - fz * eyeZ);
}

// 生成汽车引擎盖网格（位置、法线、UV）
static void generateHoodMesh(float width, float length, float height, int widthSegments, int lengthSegments,
                             GLfloat **outVertices, GLushort **outIndices,
                             int *outVertexCount, int *outIndexCount) {
    int vertexCount = (widthSegments + 1) * (lengthSegments + 1);
    int indexCount = widthSegments * lengthSegments * 6;

    // 每个顶点包含：位置(3) + 法线(3) + UV(2) = 8个float
    GLfloat *vertices = (GLfloat *)malloc(sizeof(GLfloat) * vertexCount * 8);
    GLushort *indices = (GLushort *)malloc(sizeof(GLushort) * indexCount);

    int vertexIndex = 0;
    for (int i = 0; i <= widthSegments; ++i) {
        float u = (float)i / (float)widthSegments; // 0..1
        float x = (u - 0.5f) * width; // -width/2 .. width/2
        
        for (int j = 0; j <= lengthSegments; ++j) {
            float v = (float)j / (float)lengthSegments; // 0..1
            float z = (v - 0.5f) * length; // -length/2 .. length/2
            
            // 计算引擎盖的高度 - 创建椭圆形凹陷
            float distanceFromCenter = sqrtf(x * x / (width * width * 0.25f) + z * z / (length * length * 0.25f));
            float hoodHeight = height * (1.0f - distanceFromCenter * 0.3f); // 中心凹陷
            
            // 添加边缘圆滑过渡
            if (distanceFromCenter > 0.8f) {
                hoodHeight *= (1.0f - distanceFromCenter) * 5.0f; // 边缘逐渐降低
            }
            
            // 添加一些随机的不规则性，模拟真实引擎盖
            float noise = sinf(x * 0.1f) * cosf(z * 0.1f) * 0.02f;
            hoodHeight += noise;
            
            // 位置
            vertices[vertexIndex++] = x;
            vertices[vertexIndex++] = hoodHeight;
            vertices[vertexIndex++] = z;
            
            // 计算法线 - 通过相邻顶点计算
            float epsilon = 0.01f;
            float h1, h2, h3, h4;
            
            // 计算相邻点的高度
            float x1 = x + epsilon, z1 = z;
            float x2 = x - epsilon, z2 = z;
            float x3 = x, z3 = z + epsilon;
            float x4 = x, z4 = z - epsilon;
            
            // 重新计算高度
            float d1 = sqrtf(x1 * x1 / (width * width * 0.25f) + z1 * z1 / (length * length * 0.25f));
            h1 = height * (1.0f - d1 * 0.3f);
            if (d1 > 0.8f) h1 *= (1.0f - d1) * 5.0f;
            h1 += sinf(x1 * 0.1f) * cosf(z1 * 0.1f) * 0.02f;
            
            float d2 = sqrtf(x2 * x2 / (width * width * 0.25f) + z2 * z2 / (length * length * 0.25f));
            h2 = height * (1.0f - d2 * 0.3f);
            if (d2 > 0.8f) h2 *= (1.0f - d2) * 5.0f;
            h2 += sinf(x2 * 0.1f) * cosf(z2 * 0.1f) * 0.02f;
            
            float d3 = sqrtf(x3 * x3 / (width * width * 0.25f) + z3 * z3 / (length * length * 0.25f));
            h3 = height * (1.0f - d3 * 0.3f);
            if (d3 > 0.8f) h3 *= (1.0f - d3) * 5.0f;
            h3 += sinf(x3 * 0.1f) * cosf(z3 * 0.1f) * 0.02f;
            
            float d4 = sqrtf(x4 * x4 / (width * width * 0.25f) + z4 * z4 / (length * length * 0.25f));
            h4 = height * (1.0f - d4 * 0.3f);
            if (d4 > 0.8f) h4 *= (1.0f - d4) * 5.0f;
            h4 += sinf(x4 * 0.1f) * cosf(z4 * 0.1f) * 0.02f;
            
            // 计算法线向量
            float nx = (h1 - h2) / (2.0f * epsilon);
            float nz = (h3 - h4) / (2.0f * epsilon);
            float ny = 1.0f;
            
            // 归一化法线
            float length = sqrtf(nx * nx + ny * ny + nz * nz);
            if (length > 0.0f) {
                nx /= length;
                ny /= length;
                nz /= length;
            }
            
            vertices[vertexIndex++] = nx;
            vertices[vertexIndex++] = ny;
            vertices[vertexIndex++] = nz;
            
            // UV坐标
            vertices[vertexIndex++] = u;
            vertices[vertexIndex++] = v;
        }
    }

    int idx = 0;
    for (int i = 0; i < widthSegments; ++i) {
        for (int j = 0; j < lengthSegments; ++j) {
            int row1 = i * (lengthSegments + 1);
            int row2 = (i + 1) * (lengthSegments + 1);

            GLushort a = (GLushort)(row1 + j);
            GLushort b = (GLushort)(row2 + j);
            GLushort c = (GLushort)(row2 + j + 1);
            GLushort d = (GLushort)(row1 + j + 1);

            indices[idx++] = a; indices[idx++] = b; indices[idx++] = c;
            indices[idx++] = a; indices[idx++] = c; indices[idx++] = d;
        }
    }

    *outVertices = vertices;
    *outIndices = indices;
    *outVertexCount = vertexCount;
    *outIndexCount = indexCount;
}

// Load shader helper function
static GLuint loadShader(GLenum type, const char* shaderCode) {
    GLuint shader = glCreateShader(type);
    glShaderSource(shader, 1, &shaderCode, NULL);
    glCompileShader(shader);
    
    // Check compilation status
    GLint compiled = 0;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &compiled);
    if (!compiled) {
        GLint infoLen = 0;
        glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1) {
            char* infoLog = (char*)malloc(sizeof(char) * infoLen);
            glGetShaderInfoLog(shader, infoLen, NULL, infoLog);
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Error compiling shader: %s", infoLog);
            free(infoLog);
        }
        glDeleteShader(shader);
        return 0;
    } else {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Shader compiled successfully");
    }
    
    return shader;
}

// Create shader program
static void createProgram() {
    if (program) {
        glDeleteProgram(program);
        program = 0;
    }
    GLint maxVertexAttribs;
    glGetIntegerv(GL_MAX_VERTEX_ATTRIBS, &maxVertexAttribs);
    GLuint vertexShader = loadShader(GL_VERTEX_SHADER, vertexShaderCode);
    if (!vertexShader) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to load vertex shader");
        return;
    }
    
    GLuint fragmentShader = loadShader(GL_FRAGMENT_SHADER, fragmentShaderCode);
    if (!fragmentShader) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to load fragment shader");
        glDeleteShader(vertexShader);
        return;
    }
    
    program = glCreateProgram();
    if (!program) {
        __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to create program");
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return;
    }
    
    glAttachShader(program, vertexShader);
    glAttachShader(program, fragmentShader);
    glLinkProgram(program);
    
    // Check linking status
    GLint linked = 0;
    glGetProgramiv(program, GL_LINK_STATUS, &linked);
    if (!linked) {
        GLint infoLen = 0;
        glGetProgramiv(program, GL_INFO_LOG_LENGTH, &infoLen);
        if (infoLen > 1) {
            char* infoLog = (char*)malloc(sizeof(char) * infoLen);
            glGetProgramInfoLog(program, infoLen, NULL, infoLog);
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Error linking program: %s", infoLog);
            free(infoLog);
        }
        glDeleteProgram(program);
        program = 0;
        glDeleteShader(vertexShader);
        glDeleteShader(fragmentShader);
        return;
    }
    
    // Get attribute locations
    positionHandle = glGetAttribLocation(program, "aPosition");
    normalHandle = glGetAttribLocation(program, "aNormal");
    uv0Handle = glGetAttribLocation(program, "aUV0");
    
    // 获取矩阵 uniform 变量位置
    uModelMatrixLocation = glGetUniformLocation(program, "uModelMatrix");
    uViewMatrixLocation = glGetUniformLocation(program, "uViewMatrix");
    uProjectionMatrixLocation = glGetUniformLocation(program, "uProjectionMatrix");
    uNormalMatrixLocation = glGetUniformLocation(program, "uNormalMatrix");
    
    // 获取 uniform 变量位置（按照 Filament 标准）
    uCameraPositionLocation = glGetUniformLocation(program, "uCameraPosition");
    uLightDirectionLocation = glGetUniformLocation(program, "uLightDirection");
    uLightColorIntensityLocation = glGetUniformLocation(program, "uLightColorIntensity");
    uBaseColorFactorLocation = glGetUniformLocation(program, "uBaseColorFactor");
    uMetallicFactorLocation = glGetUniformLocation(program, "uMetallicFactor");
    uRoughnessFactorLocation = glGetUniformLocation(program, "uRoughnessFactor");
    uNormalScaleLocation = glGetUniformLocation(program, "uNormalScale");
    uAOStrengthLocation = glGetUniformLocation(program, "uAOStrength");
    uEmissiveFactorLocation = glGetUniformLocation(program, "uEmissiveFactor");
    uEmissiveStrengthLocation = glGetUniformLocation(program, "uEmissiveStrength");
    uSpecularStrengthLocation = glGetUniformLocation(program, "uSpecularStrength");
    uSpecularColorFactorLocation = glGetUniformLocation(program, "uSpecularColorFactor");
    
    // Clean up shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

extern "C" {
    // Asset Manager 初始化
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_initAssetManager(JNIEnv *env, jclass clazz, jobject assetManager) {
        g_assetManager = AAssetManager_fromJava(env, assetManager);
        __android_log_print(ANDROID_LOG_INFO, TAG, "Asset manager initialized");
    }
    
    
    // 触摸事件处理函数
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_onTouchDown(JNIEnv *env, jclass clazz, jfloat x, jfloat y) {
        lastTouchX = x;
        lastTouchY = y;
        isTouching = true;
        isTwoFinger = false;
        __android_log_print(ANDROID_LOG_INFO, TAG, "Touch down: %f, %f", x, y);
    }
    
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_onTouchMove(JNIEnv *env, jclass clazz, jfloat x, jfloat y) {
        if (!isTouching) return;
        
        float deltaX = x - lastTouchX;
        float deltaY = y - lastTouchY;
        
        // 根据滑动距离更新旋转角度 - 调整灵敏度
        rotationY += deltaX * 0.01f; // 水平滑动控制Y轴旋转
        rotationX += deltaY * 0.01f; // 垂直滑动控制X轴旋转
        
        // 限制X轴旋转角度
        if (rotationX > M_PI / 2) rotationX = M_PI / 2;
        if (rotationX < -M_PI / 2) rotationX = -M_PI / 2;
        
        lastTouchX = x;
        lastTouchY = y;
        
        __android_log_print(ANDROID_LOG_INFO, TAG, "Touch move: rotationX=%f, rotationY=%f", rotationX, rotationY);
    }
    
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_onTouchUp(JNIEnv *env, jclass clazz) {
        isTouching = false;
        isTwoFinger = false;
        __android_log_print(ANDROID_LOG_INFO, TAG, "Touch up");
    }
    
    // 双指触摸事件处理函数
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_onTwoFingerDown(JNIEnv *env, jclass clazz, jfloat x1, jfloat y1, jfloat x2, jfloat y2) {
        lastTouchX = (x1 + x2) / 2.0f;
        lastTouchY = (y1 + y2) / 2.0f;
        lastDistance = calculateDistance(x1, y1, x2, y2);
        isTouching = true;
        isTwoFinger = true;
        __android_log_print(ANDROID_LOG_INFO, TAG, "Two finger down: (%f,%f) (%f,%f), distance=%f", x1, y1, x2, y2, lastDistance);
    }
    
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_onTwoFingerMove(JNIEnv *env, jclass clazz, jfloat x1, jfloat y1, jfloat x2, jfloat y2) {
        if (!isTouching || !isTwoFinger) return;
        
        float currentDistance = calculateDistance(x1, y1, x2, y2);
        if (lastDistance > 0.0f) {
            float scaleFactor = currentDistance / lastDistance;
            scale *= scaleFactor;
            
            // 限制缩放范围
            if (scale < 0.1f) scale = 0.1f;
            if (scale > 10.0f) scale = 10.0f;
            
            __android_log_print(ANDROID_LOG_INFO, TAG, "Two finger move: scale=%f", scale);
        }
        
        lastDistance = currentDistance;
        lastTouchX = (x1 + x2) / 2.0f;
        lastTouchY = (y1 + y2) / 2.0f;
    }
    
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_onTwoFingerUp(JNIEnv *env, jclass clazz) {
        isTouching = false;
        isTwoFinger = false;
        __android_log_print(ANDROID_LOG_INFO, TAG, "Two finger up");
    }

    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_init(JNIEnv *env, jclass clazz) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Initializing TriangleRenderer");
        createProgram();
        
        // 检查Asset Manager是否已初始化
        if (!g_assetManager) {
            __android_log_print(ANDROID_LOG_ERROR, TAG, "Asset Manager not initialized! Call initAssetManager first.");
            return;
        }
        
        // 加载OBJ文件
        if (vbo == 0 || ibo == 0) {
            g_objMesh = loadOBJFile("mesh.obj"); // 假设OBJ文件名为model.obj
            
            if (g_objMesh.vertexCount > 0) {
                // 创建顶点数据数组
                GLfloat *vertices = (GLfloat*)malloc(sizeof(GLfloat) * g_objMesh.vertexCount * 8);
                GLushort *indices = (GLushort*)malloc(sizeof(GLushort) * g_objMesh.indexCount);
                
                // 填充顶点数据
                for (int i = 0; i < g_objMesh.vertexCount; i++) {
                    int baseIdx = i * 8;
                    vertices[baseIdx + 0] = g_objMesh.vertices[i].x;
                    vertices[baseIdx + 1] = g_objMesh.vertices[i].y;
                    vertices[baseIdx + 2] = g_objMesh.vertices[i].z;
                    vertices[baseIdx + 3] = g_objMesh.vertices[i].nx;
                    vertices[baseIdx + 4] = g_objMesh.vertices[i].ny;
                    vertices[baseIdx + 5] = g_objMesh.vertices[i].nz;
                    vertices[baseIdx + 6] = g_objMesh.vertices[i].u;
                    vertices[baseIdx + 7] = g_objMesh.vertices[i].v;
                }
                
                // 填充索引数据
                for (int i = 0; i < g_objMesh.indexCount; i++) {
                    indices[i] = g_objMesh.indices[i];
                }
                
                if (vbo == 0) glGenBuffers(1, &vbo);
                glBindBuffer(GL_ARRAY_BUFFER, vbo);
                glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * g_objMesh.vertexCount * 8, vertices, GL_STATIC_DRAW);

                if (ibo == 0) glGenBuffers(1, &ibo);
                glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * g_objMesh.indexCount, indices, GL_STATIC_DRAW);

                objIndexCount = (GLsizei)g_objMesh.indexCount;

                free(vertices);
                free(indices);
                
                __android_log_print(ANDROID_LOG_INFO, TAG, "OBJ mesh loaded successfully");
            } else {
                __android_log_print(ANDROID_LOG_ERROR, TAG, "Failed to load OBJ mesh, using fallback");
                // 如果OBJ加载失败，可以在这里添加fallback逻辑
            }
        }
    }

    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_surfaceCreated(JNIEnv *env, jclass clazz) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Surface created");
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // 设置背景色为黑色
        glEnable(GL_DEPTH_TEST);
    }

    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_surfaceChanged(JNIEnv *env, jclass clazz, jint width, jint height) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Surface changed: %d x %d", width, height);
        glViewport(0, 500, width, width);
        
        // 存储屏幕尺寸用于投影矩阵计算
        static int screenWidth = width;
        static int screenHeight = height;
        screenWidth = width;
        screenHeight = height;
    }

    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_drawFrame(JNIEnv *env, jclass clazz) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (program) {
            glUseProgram(program);
            
            __android_log_print(ANDROID_LOG_INFO, TAG, "Drawing frame with PBR shader");
            
            // 创建变换矩阵
            float modelMatrix[16];
            float viewMatrix[16];
            float projectionMatrix[16];
            float mvpMatrix[16];
            
            // 创建模型矩阵（先缩放，再旋转，确保围绕中心点旋转）
            createIdentityMatrix(modelMatrix);
            
            // 1. 先应用缩放（基础缩放 + 用户缩放）
            float baseScale = 20.0f;
            float totalScale = baseScale * scale;
            float scaleMatrix[16];
            createIdentityMatrix(scaleMatrix);
            scaleMatrix[0] = totalScale;  // X轴缩放
            scaleMatrix[5] = totalScale;  // Y轴缩放
            scaleMatrix[10] = totalScale; // Z轴缩放
            
            // 2. 再应用旋转
            float rotationMatrix[16];
            createRotationMatrix(rotationMatrix, rotationX, rotationY);
            
            // 3. 组合矩阵：Model = Rotation * Scale
            float tempMatrix[16];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    tempMatrix[i * 4 + j] = 0.0f;
                    for (int k = 0; k < 4; k++) {
                        tempMatrix[i * 4 + j] += rotationMatrix[i * 4 + k] * scaleMatrix[k * 4 + j];
                    }
                }
            }
            memcpy(modelMatrix, tempMatrix, 16 * sizeof(float));
            
            // 创建视图矩阵 - 调整相机位置使模型更大
            createViewMatrix(viewMatrix, 0.0f, 0.0f, 1.5f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);
            
            // 创建正交投影矩阵 - 消除近大远小的透视效果
            static float aspect = 1.0f;
            static int lastWidth = 0, lastHeight = 0;
            // 使用正交投影，设置合适的视口范围
            createOrthographicMatrix(projectionMatrix, -2.0f, 2.0f, -2.0f, 2.0f, 0.01f, 10.0f);

            
            // 设置矩阵uniform
            if (uModelMatrixLocation >= 0) {
                glUniformMatrix4fv(uModelMatrixLocation, 1, GL_FALSE, modelMatrix);
            }
            if (uViewMatrixLocation >= 0) {
                glUniformMatrix4fv(uViewMatrixLocation, 1, GL_FALSE, viewMatrix);
            }
            if (uProjectionMatrixLocation >= 0) {
                glUniformMatrix4fv(uProjectionMatrixLocation, 1, GL_FALSE, projectionMatrix);
            }
            
            // 设置光照参数 - 调整以获得更好的明暗对比
            if (uCameraPositionLocation >= 0) {
                glUniform3f(uCameraPositionLocation, 0.0f, 0.0f, 1.5f); // 与视图矩阵保持一致
            }
            if (uLightDirectionLocation >= 0) {
                glUniform3f(uLightDirectionLocation, 0.3f, 0.8f, -0.5f); // 调整光照方向以获得更好的阴影
            }
            if (uLightColorIntensityLocation >= 0) {
                glUniform4f(uLightColorIntensityLocation, 1.0f, 1.0f, 1.0f, 2.0f); // 降低光照强度
            }
            
            // 设置材质参数（改进的材质设置）
            if (uBaseColorFactorLocation >= 0) {
                glUniform4f(uBaseColorFactorLocation, 0.6f, 0.6f, 0.6f, 1.0f); // 适中的灰白色
            }
            if (uMetallicFactorLocation >= 0) {
                glUniform1f(uMetallicFactorLocation, 0.3f); // 增加金属感以获得更好的反射
            }
            if (uRoughnessFactorLocation >= 0) {
                glUniform1f(uRoughnessFactorLocation, 0.2f); // 稍微降低粗糙度以获得更清晰的高光
            }
            if (uNormalScaleLocation >= 0) {
                glUniform1f(uNormalScaleLocation, 1.0f); // 法线强度
            }
            if (uAOStrengthLocation >= 0) {
                glUniform1f(uAOStrengthLocation, 1.0f); // 环境光遮蔽
            }
            if (uEmissiveFactorLocation >= 0) {
                glUniform3f(uEmissiveFactorLocation, 0.0f, 0.0f, 0.0f); // 自发光颜色
            }
            if (uEmissiveStrengthLocation >= 0) {
                glUniform1f(uEmissiveStrengthLocation, 1.0f); // 自发光强度
            }
            if (uSpecularStrengthLocation >= 0) {
                glUniform1f(uSpecularStrengthLocation, 1.0f); // 镜面反射强度
            }
            if (uSpecularColorFactorLocation >= 0) {
                glUniform3f(uSpecularColorFactorLocation, 1.0f, 1.0f, 1.0f); // 镜面反射颜色
            }
            
            // Set up vertex attributes
            glEnableVertexAttribArray(positionHandle);
            glEnableVertexAttribArray(normalHandle);
            glEnableVertexAttribArray(uv0Handle);
            
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexAttribPointer(positionHandle, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)0);
            glVertexAttribPointer(normalHandle, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(3 * sizeof(GLfloat)));
            glVertexAttribPointer(uv0Handle, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (void*)(6 * sizeof(GLfloat)));
            
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
            glDrawElements(GL_TRIANGLES, objIndexCount, GL_UNSIGNED_SHORT, (void*)0);
            
            glDisableVertexAttribArray(positionHandle);
            glDisableVertexAttribArray(normalHandle);
            glDisableVertexAttribArray(uv0Handle);
        }
    }
}