import {BrowserRouter} from 'react-router-dom';

import {ErrorBoundary} from '../components/ErrorBoundary';
import {AppRoutes} from './routes';

export function App() {
  return (
    <ErrorBoundary>
      <BrowserRouter>
        <AppRoutes />
      </BrowserRouter>
    </ErrorBoundary>
  );
}
