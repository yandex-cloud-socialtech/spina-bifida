import '@gravity-ui/uikit/styles/styles.css';
import './i18n';
import './styles/tokens.css';
import './styles/global.css';

import {StrictMode} from 'react';
import {createRoot} from 'react-dom/client';

import {App} from './app/App';
import {AppProviders} from './app/providers';

const root = document.getElementById('root');

if (!root) {
  throw new Error('Application root is missing');
}

createRoot(root).render(
  <StrictMode>
    <AppProviders>
      <App />
    </AppProviders>
  </StrictMode>,
);
