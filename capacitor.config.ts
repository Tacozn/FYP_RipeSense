import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  "appId": "com.ripesense.app",
  "appName": "RipeSense",
  "webDir": "static",
  "server": {
    "url": "https://ripesense-production.up.railway.app",
    "cleartext": true
  },
  "android": {
    "allowMixedContent": true
  }
};

export default config;
