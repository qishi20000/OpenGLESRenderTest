package com.example.openglestriangle;

import android.opengl.GLSurfaceView;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class TriangleRenderer implements GLSurfaceView.Renderer {
    static {
        System.loadLibrary("triangle_renderer");
    }

    // Native methods
    public static native void initAssetManager(android.content.res.AssetManager assetManager);
    public static native void init();
    public static native void surfaceCreated();
    public static native void surfaceChanged(int width, int height);
    public static native void drawFrame();
    
    // Touch event methods
    public static native void onTouchDown(float x, float y);
    public static native void onTouchMove(float x, float y);
    public static native void onTouchUp();

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        init();
        surfaceCreated();
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        surfaceChanged(width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        drawFrame();
    }
}