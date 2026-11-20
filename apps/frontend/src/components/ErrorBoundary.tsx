import {Button, Card} from '@gravity-ui/uikit';
import {Component, type ErrorInfo, type ReactNode} from 'react';

import i18n from '../i18n';

interface Props {
  children: ReactNode;
}

interface State {
  failed: boolean;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = {failed: false};

  static getDerivedStateFromError(): State {
    return {failed: true};
  }

  componentDidCatch(error: Error, info: ErrorInfo): void {
    console.error('React render failed', error, info.componentStack);
  }

  render() {
    if (!this.state.failed) {
      return this.props.children;
    }
    return (
      <main className="screen-status screen-status_error">
        <Card type="container" view="outlined" theme="danger" size="l">
          <p>{i18n.t('errors.unknown')}</p>
          <Button onClick={() => window.location.reload()}>{i18n.t('common.retry')}</Button>
        </Card>
      </main>
    );
  }
}
