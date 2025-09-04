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
        void main() {
          vNormal = normalize(aPosition);
          gl_Position = vec4(aPosition, 1.0);
        }
        )";

static const char* fragmentShaderCode = R"(
        #version 300 es
        precision mediump float;
        in vec3 vNormal;
        uniform vec3 uLightDir;
        out vec4 fragColor;
        void main() {
          vec3 n = normalize(vNormal);
          vec3 l = normalize(uLightDir);
          float diff = max(dot(n, l), 0.0);
          vec3 base = vec3(0.8, 0.8, 0.8);
          vec3 color = base * (0.2 + 0.8 * diff);
          fragColor = vec4(color, 1.0);
        }
        )";

static GLuint program = 0;
static GLint positionHandle = 0;
static GLint uLightDirLocation = -1;
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
            // 设置方向光（世界坐标等于物体坐标，此处球心在原点）
            if (uLightDirLocation >= 0) {
                glUniform3f(uLightDirLocation, 0.5f, 1.0f, -1.3f);
            }
            glEnableVertexAttribArray(positionHandle);
            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glVertexAttribPointer(positionHandle, 3, GL_FLOAT, GL_FALSE, 0, (void*)0);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
            glDrawElements(GL_TRIANGLES, sphereIndexCount, GL_UNSIGNED_SHORT, (void*)0);
            glDisableVertexAttribArray(positionHandle);
        }
    }
}