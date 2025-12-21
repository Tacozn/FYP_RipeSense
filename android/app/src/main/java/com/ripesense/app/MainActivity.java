package com.ripesense.app;

import com.getcapacitor.BridgeActivity;
import android.os.Bundle;

public class MainActivity extends BridgeActivity {

    @Override
    public void onBackPressed() {
        // [DETECTIVE] Lee Wonhee's "Back Button" Logic

        // 1. Get the current URL the user is looking at
        String currentUrl = bridge.getWebView().getUrl();

        // 2. Define your Home URL (UPDATE THIS!)
        String homeUrl = "https://ripesense-production.up.railway.app/";

        // 3. The Logic:
        if (currentUrl != null && !currentUrl.equals(homeUrl)) {
            // If we are NOT on the home page, force the browser to go Home
            bridge.getWebView().loadUrl(homeUrl);
        } else {
            // If we ARE on the home page, behave normally (close the app)
            super.onBackPressed();
        }
    }
}
