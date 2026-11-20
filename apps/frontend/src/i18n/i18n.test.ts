import {fireEvent, render, screen, waitFor} from '@testing-library/react';
import {createElement} from 'react';
import {describe, expect, it, vi} from 'vitest';

import {apiClient} from '../api/client';
import {LocaleSelector} from '../components/AppHeader';
import i18n from './index';
import en from './en.json';
import kk from './kk.json';
import ru from './ru.json';
import uz from './uz.json';

function flatten(value: Record<string, unknown>, prefix = ''): string[] {
  return Object.entries(value).flatMap(([key, child]) => {
    const path = prefix ? `${prefix}.${key}` : key;
    return child && typeof child === 'object' && !Array.isArray(child)
      ? flatten(child as Record<string, unknown>, path)
      : [path];
  });
}

function leafValues(value: Record<string, unknown>): string[] {
  return Object.values(value).flatMap((child) =>
    child && typeof child === 'object' && !Array.isArray(child)
      ? leafValues(child as Record<string, unknown>)
      : [String(child)],
  );
}

describe('locales', () => {
  it('keeps exact key parity and safe medical wording', () => {
    const sourceKeys = flatten(ru).sort();
    for (const locale of [en, kk, uz]) {
      expect(flatten(locale).sort()).toEqual(sourceKeys);
    }
    for (const locale of [ru, en, kk, uz]) {
      const copy = JSON.stringify(locale).toLocaleLowerCase();
      expect(copy).toContain('spina bifida');
      for (const forbidden of [
        'диагноз подтверждён',
        'diagnosis confirmed',
        'диагноз расталды',
        'tashxis tasdiqlandi',
      ]) {
        expect(copy).not.toContain(forbidden);
      }
    }
    expect(ru.analysis.upload).toBe('Загрузить изображения');
    expect(ru.analysis.loadExamples).toBe('Открыть примеры');
    expect('exampleHint' in ru.analysis).toBe(false);
  });

  it('does not restore the removed legends, technical panel or old quality guidance copy', () => {
    const removedKeys = [
      'analysis.replaceFrame',
      'analysis.startNew',
      'analysis.layers.roiLegend',
      'analysis.layers.heatmapLegend',
      'analysis.result.roiHeading',
      'analysis.result.explanationHeading',
      'analysis.result.actionsHeading',
      'analysis.result.probabilityLabel',
      'analysis.result.modelDetails',
      'analysis.warning',
      'medical.qualityAccepted',
      'medical.qualityRejected',
    ];
    const removedCopy = [
      'Контур показывает найденную моделью область интереса.',
      'The outline marks the region of interest located by the model.',
      'Контур модель тапқан қызығушылық аймағын көрсетеді.',
      'Kontur model topgan qiziqish sohasini ko‘rsatadi.',
      'Результат ИИ не является медицинским заключением. Сервис предназначен только для исследовательского использования.',
      'The AI result is not a medical conclusion. The service is intended for research use only.',
      'ЖИ нәтижесі медициналық қорытынды болып табылмайды. Сервис тек зерттеу мақсатында пайдалануға арналған.',
      'SI natijasi tibbiy xulosa hisoblanmaydi. Xizmat faqat tadqiqot maqsadida foydalanish uchun mo‘ljallangan.',
      'Качество изображения не позволяет выполнить анализ',
      'Image quality is insufficient for analysis',
      'Кескін сапасы талдау үшін жеткіліксіз',
      'Tasvir sifati tahlil uchun yetarli emas',
    ];

    for (const locale of [ru, en, kk, uz]) {
      const keys = flatten(locale);
      const values = leafValues(locale);
      for (const key of removedKeys) {
        expect(keys).not.toContain(key);
      }
      for (const copy of removedCopy) {
        expect(values).not.toContain(copy);
      }
    }
  });

  it('updates the preference without remounting the current analysis route', async () => {
    const request = vi.spyOn(apiClient, 'request').mockResolvedValue({
      user: {id: 'u1', email: 'doctor@example.test', role: 'user', status: 'active', locale: 'en'},
    });
    let mounts = 0;
    function CurrentAnalysis() {
      mounts += 1;
      return createElement('div', null, 'analysis-j1');
    }
    render(
      createElement(
        'div',
        null,
        createElement(LocaleSelector, {currentLocale: 'ru'}),
        createElement(CurrentAnalysis),
      ),
    );

    fireEvent.change(screen.getByRole('combobox'), {target: {value: 'en'}});

    await waitFor(() =>
      expect(request).toHaveBeenCalledWith('/api/v1/auth/preferences', {
        method: 'PATCH',
        json: {locale: 'en'},
      }),
    );
    expect(mounts).toBe(1);
    expect(localStorage.getItem('locale')).toBe('en');
    request.mockRestore();
  });

  it('restores the current locale after a failed preference update', async () => {
    localStorage.setItem('locale', 'ru');
    await i18n.changeLanguage('en');
    const request = vi.spyOn(apiClient, 'request').mockRejectedValue(new Error('network'));
    render(createElement(LocaleSelector, {currentLocale: 'ru'}));

    fireEvent.change(screen.getByRole('combobox'), {target: {value: 'en'}});

    await waitFor(() => expect(i18n.language).toBe('ru'));
    expect(screen.getByRole('combobox', {name: 'Язык'})).toHaveValue('ru');
    expect(localStorage.getItem('locale')).toBe('ru');
    request.mockRestore();
  });
});
