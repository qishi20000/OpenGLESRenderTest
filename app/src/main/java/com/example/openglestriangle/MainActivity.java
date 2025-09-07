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
        public CustomGLSurfaceView(Context context) {
            super(context);
        }
        
        @Override
        public boolean onTouchEvent(MotionEvent event) {
            float x = event.getX();
            float y = event.getY();
            
            switch (event.getAction()) {
                case MotionEvent.ACTION_DOWN:
                    TriangleRenderer.onTouchDown(x, y);
                    return true;
                case MotionEvent.ACTION_MOVE:
                    TriangleRenderer.onTouchMove(x, y);
                    return true;
                case MotionEvent.ACTION_UP:
                case MotionEvent.ACTION_CANCEL:
                    TriangleRenderer.onTouchUp();
                    return true;
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