#include <jni.h>
#include <android/log.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <stdlib.h>  // 添加此头文件以使用malloc和free
#include <math.h>

static const char* TAG = "OpenGLESTriangle";

// Shader sources
static const char* vertexShaderCode = R"(
        #version 300 es
        layout(location = 0) in vec3 aPosition;
        out vec3 vNormal;
        out vec3 vWorldPos;
        void main() {
          vNormal = normalize(aPosition);
          vWorldPos = aPosition;
          gl_Position = vec4(aPosition, 1.0);
        }
        )";

static const char* fragmentShaderCode = R"(
        #version 300 es
        precision mediump float;
        in vec3 vNormal;
        in vec3 vWorldPos;
        
        // Directional light
        uniform vec3 uLightDir;           // light direction (towards light)
        uniform vec3 uLightColor;         // light color (RGB)
        uniform float uLightIntensity;    // scalar intensity
        
        // Camera
        uniform vec3 uCameraPos;          // camera position in world space
        
        // PBR material params
        uniform vec3 uBaseColor;          // base color / albedo
        uniform float uMetallic;          // [0,1]
        uniform float uRoughness;         // [0.04,1]
        uniform float uAmbientOcclusion;  // [0,1]
        
        // Clearcoat (second specular lobe similar to Filament)
        uniform float uClearcoat;         // [0,1]
        uniform float uClearcoatRoughness;// [0,1]
        
        out vec4 fragColor;
        
        // Schlick Fresnel approximation
        vec3 fresnelSchlick(float cosTheta, vec3 F0) {
          return F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
        }
        
        float distributionGGX(float NdotH, float alpha) {
          float a2 = alpha * alpha;
          float denom = (NdotH * NdotH) * (a2 - 1.0) + 1.0;
          return a2 / (3.14159265 * denom * denom);
        }
        
        float geometrySmithGGX(float NdotV, float NdotL, float alpha) {
          // Schlick-GGX masking-shadowing approximation
          float k = (alpha + 1.0);
          k = (k * k) / 8.0;
          float gV = NdotV / (NdotV * (1.0 - k) + k);
          float gL = NdotL / (NdotL * (1.0 - k) + k);
          return gV * gL;
        }
        
        void main() {
          // Normal, Light, View, Half
          vec3 N = normalize(vNormal);
          vec3 L = normalize(uLightDir);
          vec3 V = normalize(uCameraPos - vWorldPos);
          vec3 H = normalize(V + L);
          
          float NdotL = max(dot(N, L), 0.0);
          float NdotV = max(dot(N, V), 1e-4);
          float NdotH = max(dot(N, H), 0.0);
          float VdotH = max(dot(V, H), 0.0);
          
          // Base reflectance F0
          vec3 dielectricF0 = vec3(0.04);
          vec3 F0 = mix(dielectricF0, uBaseColor, uMetallic);
          
          float alpha = max(uRoughness * uRoughness, 1e-4);
          
          // Cook-Torrance BRDF
          float D = distributionGGX(NdotH, alpha);
          float G = geometrySmithGGX(NdotV, NdotL, alpha);
          vec3  F = fresnelSchlick(VdotH, F0);
          
          vec3 numerator = D * G * F;
          float denom = 4.0 * NdotV * NdotL + 1e-4;
          vec3 specular = numerator / denom;
          
          // Energy-conserving diffuse term
          vec3 kS = F;                      // specular amount
          vec3 kD = (vec3(1.0) - kS) * (1.0 - uMetallic);
          vec3 diffuse = kD * uBaseColor / 3.14159265;
          
          // Clearcoat lobe (thin dielectric layer)
          float ccAlpha = max(uClearcoatRoughness * uClearcoatRoughness, 1e-4);
          float Dcc = distributionGGX(NdotH, ccAlpha);
          float Gcc = geometrySmithGGX(NdotV, NdotL, ccAlpha);
          float Fcc = fresnelSchlick(VdotH, vec3(0.04)).r; // scalar F for clearcoat
          float ccDenom = 4.0 * NdotV * NdotL + 1e-4;
          float clearcoatSpec = (Dcc * Gcc * Fcc) / ccDenom;
          
          // Direct lighting
          vec3 radiance = uLightColor * uLightIntensity;
          vec3 direct = (diffuse + specular + uClearcoat * clearcoatSpec) * radiance * NdotL;
          
          // Simple ambient (AO)
          vec3 ambient = uBaseColor * 0.03 * uAmbientOcclusion; // small ambient term
          
          vec3 color = ambient + direct;
          fragColor = vec4(color, 1.0);
        }
        )";

static GLuint program = 0;
static GLint positionHandle = 0;
static GLint uLightDirLocation = -1;
static GLint uLightColorLocation = -1;
static GLint uLightIntensityLocation = -1;
static GLint uCameraPosLocation = -1;
static GLint uBaseColorLocation = -1;
static GLint uMetallicLocation = -1;
static GLint uRoughnessLocation = -1;
static GLint uAOLocation = -1;
static GLint uClearcoatLocation = -1;
static GLint uClearcoatRoughnessLocation = -1;
GLuint vbo = 0;
GLuint ibo = 0;
GLsizei sphereIndexCount = 0;

// 生成球体网格（仅位置）
static void generateSphereMesh(float radius, int stacks, int slices,
                               GLfloat **outVertices, GLushort **outIndices,
                               int *outVertexCount, int *outIndexCount) {
    int vertexCount = (stacks + 1) * (slices + 1);
    int indexCount = stacks * slices * 6;

    GLfloat *vertices = (GLfloat *)malloc(sizeof(GLfloat) * vertexCount * 3);
    GLushort *indices = (GLushort *)malloc(sizeof(GLushort) * indexCount);

    int v = 0;
    for (int i = 0; i <= stacks; ++i) {
        float vRatio = (float)i / (float)stacks;
        float phi = (float)M_PI * vRatio;    // 0..PI
        float y = cosf(phi);
        float r = sinf(phi);

        for (int j = 0; j <= slices; ++j) {
            float uRatio = (float)j / (float)slices;
            float theta = 2.0f * (float)M_PI * uRatio; // 0..2PI
            float x = r * cosf(theta);
            float z = r * sinf(theta);

            vertices[v++] = radius * x;
            vertices[v++] = radius * y;
            vertices[v++] = radius * z;
        }
    }

    int idx = 0;
    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int row1 = i * (slices + 1);
            int row2 = (i + 1) * (slices + 1);

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
    
    positionHandle = glGetAttribLocation(program, "aPosition");
    uLightDirLocation = glGetUniformLocation(program, "uLightDir");
    uLightColorLocation = glGetUniformLocation(program, "uLightColor");
    uLightIntensityLocation = glGetUniformLocation(program, "uLightIntensity");
    uCameraPosLocation = glGetUniformLocation(program, "uCameraPos");
    uBaseColorLocation = glGetUniformLocation(program, "uBaseColor");
    uMetallicLocation = glGetUniformLocation(program, "uMetallic");
    uRoughnessLocation = glGetUniformLocation(program, "uRoughness");
    uAOLocation = glGetUniformLocation(program, "uAmbientOcclusion");
    uClearcoatLocation = glGetUniformLocation(program, "uClearcoat");
    uClearcoatRoughnessLocation = glGetUniformLocation(program, "uClearcoatRoughness");
    
    // Clean up shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

extern "C" {
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_init(JNIEnv *env, jclass clazz) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Initializing TriangleRenderer");
        createProgram();
        // 创建球体网格并上传到GPU
        if (vbo == 0 || ibo == 0) {
            GLfloat *vertices = NULL;
            GLushort *indices = NULL;
            int vertexCount = 0;
            int indexCount = 0;
            generateSphereMesh(0.7f, 40, 40, &vertices, &indices, &vertexCount, &indexCount);

            if (vbo == 0) glGenBuffers(1, &vbo);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glBufferData(GL_ARRAY_BUFFER, sizeof(GLfloat) * vertexCount * 3, vertices, GL_STATIC_DRAW);

            if (ibo == 0) glGenBuffers(1, &ibo);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(GLushort) * indexCount, indices, GL_STATIC_DRAW);

            sphereIndexCount = (GLsizei)indexCount;

            if (vertices) free(vertices);
            if (indices) free(indices);
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
    }

    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_drawFrame(JNIEnv *env, jclass clazz) {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        if (program) {
            glUseProgram(program);
            // 光照与相机
            if (uLightDirLocation >= 0) glUniform3f(uLightDirLocation, 0.5f, 1.0f, -1.3f);
            if (uLightColorLocation >= 0) glUniform3f(uLightColorLocation, 1.0f, 1.0f, 1.0f);
            if (uLightIntensityLocation >= 0) glUniform1f(uLightIntensityLocation, 2.5f);
            if (uCameraPosLocation >= 0) glUniform3f(uCameraPosLocation, 0.0f, 0.0f, 3.0f);

            // 车漆风格 PBR 参数（类似 Filament 思路：介电基底 + clearcoat）
            if (uBaseColorLocation >= 0) glUniform3f(uBaseColorLocation, 0.7f, 0.05f, 0.05f);
            if (uMetallicLocation >= 0) glUniform1f(uMetallicLocation, 0.0f);
            if (uRoughnessLocation >= 0) glUniform1f(uRoughnessLocation, 0.2f);
            if (uAOLocation >= 0) glUniform1f(uAOLocation, 1.0f);
            if (uClearcoatLocation >= 0) glUniform1f(uClearcoatLocation, 1.0f);
            if (uClearcoatRoughnessLocation >= 0) glUniform1f(uClearcoatRoughnessLocation, 0.1f);
            glEnableVertexAttribArray(positionHandle);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexAttribPointer(positionHandle, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
            glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_SHORT, (void*)0);
            glDisableVertexAttribArray(positionHandle);
        }
    }
}