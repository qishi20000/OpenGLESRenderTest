#include <jni.h>
#include <android/log.h>
#include <GLES3/gl3.h>
#include <GLES3/gl3ext.h>
#include <stdlib.h>  // 添加此头文件以使用malloc和free

static const char* TAG = "OpenGLESTriangle";

// Shader sources
static const char* vertexShaderCode =
        "#version 300 es\n"
        "layout(location = 0) in vec4 vPosition;\n"
        "void main() {\n"
        "  gl_Position = vPosition;\n"
        "}\n";

static const char* fragmentShaderCode =
        "#version 300 es\n"
        "precision mediump float;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "  fragColor = vec4(1.0, 0.0, 0.0, 1.0);\n"
        "}\n";

static GLuint program = 0;
static GLint positionHandle = 0;

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
    
    positionHandle = glGetAttribLocation(program, "vPosition");
    
    // Clean up shaders as they're linked into our program now and no longer necessary
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}

extern "C" {
    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_init(JNIEnv *env, jclass clazz) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Initializing TriangleRenderer");
        createProgram();
    }

    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_surfaceCreated(JNIEnv *env, jclass clazz) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Surface created");
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // 设置背景色为黑色
    }

    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_surfaceChanged(JNIEnv *env, jclass clazz, jint width, jint height) {
        __android_log_print(ANDROID_LOG_INFO, TAG, "Surface changed: %d x %d", width, height);
        glViewport(0, 0, width, height);
    }

    JNIEXPORT void JNICALL
    Java_com_example_openglestriangle_TriangleRenderer_drawFrame(JNIEnv *env, jclass clazz) {
        glClear(GL_COLOR_BUFFER_BIT);
        if (program) {
            glUseProgram(program);
            glEnableVertexAttribArray(positionHandle);

            static const GLfloat triangleCoords[] = {
                    0.0f,  0.5f, 0.0f,  // top
                    -0.5f, -0.5f, 0.0f,  // bottom left
                    0.5f, -0.5f, 0.0f   // bottom right
            };

            glVertexAttribPointer(positionHandle, 3, GL_FLOAT, GL_FALSE, 0, triangleCoords);
            glDrawArrays(GL_TRIANGLES, 0, 3);
            glDisableVertexAttribArray(positionHandle);
        }
    }
}