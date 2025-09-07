package com.example.openglestriangle;

import android.app.Activity;
import android.app.ActivityManager;
import android.content.Context;
import android.content.pm.ConfigurationInfo;
import android.opengl.GLSurfaceView;
import android.os.Bundle;
import android.view.MotionEvent;
import android.widget.Toast;

public class MainActivity extends Activity {
    private CustomGLSurfaceView mGLSurfaceView;
    
    // 自定义GLSurfaceView来处理触摸事件
    private class CustomGLSurfaceView extends GLSurfaceView {
        private float lastDistance = 0.0f;
        private boolean isTwoFinger = false;
        
        public CustomGLSurfaceView(Context context) {
            super(context);
        }
        
        private float calculateDistance(float x1, float y1, float x2, float y2) {
            float dx = x2 - x1;
            float dy = y2 - y1;
            return (float) Math.sqrt(dx * dx + dy * dy);
        }
        
        @Override
        public boolean onTouchEvent(MotionEvent event) {
            int pointerCount = event.getPointerCount();
            
            if (pointerCount == 1) {
                // 单指触摸 - 旋转
                float x = event.getX();
                float y = event.getY();
                
                switch (event.getAction()) {
                    case MotionEvent.ACTION_DOWN:
                        TriangleRenderer.onTouchDown(x, y);
                        isTwoFinger = false;
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        if (!isTwoFinger) {
                            TriangleRenderer.onTouchMove(x, y);
                        }
                        return true;
                    case MotionEvent.ACTION_UP:
                    case MotionEvent.ACTION_CANCEL:
                        TriangleRenderer.onTouchUp();
                        isTwoFinger = false;
                        return true;
                }
            } else if (pointerCount == 2) {
                // 双指触摸 - 缩放
                float x1 = event.getX(0);
                float y1 = event.getY(0);
                float x2 = event.getX(1);
                float y2 = event.getY(1);
                
                switch (event.getActionMasked()) {
                    case MotionEvent.ACTION_POINTER_DOWN:
                        lastDistance = calculateDistance(x1, y1, x2, y2);
                        TriangleRenderer.onTwoFingerDown(x1, y1, x2, y2);
                        isTwoFinger = true;
                        return true;
                    case MotionEvent.ACTION_MOVE:
                        if (isTwoFinger) {
                            TriangleRenderer.onTwoFingerMove(x1, y1, x2, y2);
                        }
                        return true;
                    case MotionEvent.ACTION_POINTER_UP:
                    case MotionEvent.ACTION_UP:
                    case MotionEvent.ACTION_CANCEL:
                        TriangleRenderer.onTwoFingerUp();
                        isTwoFinger = false;
                        return true;
                }
            }
            
            return super.onTouchEvent(event);
        }
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // 检查设备是否支持OpenGL ES 3.0
        ActivityManager activityManager = (ActivityManager) getSystemService(Context.ACTIVITY_SERVICE);
        ConfigurationInfo configurationInfo = activityManager.getDeviceConfigurationInfo();
        final boolean supportsEs3 = configurationInfo.reqGlEsVersion >= 0x30000;

        if (supportsEs3) {
            mGLSurfaceView = new CustomGLSurfaceView(this);
            mGLSurfaceView.setEGLContextClientVersion(3); // 使用OpenGL ES 3.0
            
            // 初始化Asset Manager
            TriangleRenderer.initAssetManager(getAssets());
            
            mGLSurfaceView.setRenderer(new TriangleRenderer());
            setContentView(mGLSurfaceView);
        } else {
            // 设备不支持OpenGL ES 3.0
            Toast.makeText(this, "设备不支持OpenGL ES 3.0", Toast.LENGTH_LONG).show();
            finish();
        }
    }

    @Override
    protected void onPause() {
        super.onPause();
        if (mGLSurfaceView != null) {
            mGLSurfaceView.onPause();
        }
    }

    @Override
    protected void onResume() {
        super.onResume();
        if (mGLSurfaceView != null) {
            mGLSurfaceView.onResume();
        }
    }
}