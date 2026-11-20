import i18n from 'i18next';
import {initReactI18next} from 'react-i18next';

import en from './en.json';
import kk from './kk.json';
import ru from './ru.json';
import uz from './uz.json';

const supported = new Set(['ru', 'en', 'kk', 'uz']);
const persisted = localStorage.getItem('locale');
const initialLanguage = persisted && supported.has(persisted) ? persisted : 'ru';

void i18n.use(initReactI18next).init({
  resources: {
    ru: {translation: ru},
    en: {translation: en},
    kk: {translation: kk},
    uz: {translation: uz},
  },
  lng: initialLanguage,
  fallbackLng: 'ru',
  interpolation: {escapeValue: false},
  returnNull: false,
});

export default i18n;
