import React, { createContext, useContext, useState, useEffect } from 'react';
import en from './locales/en.json';
import ja from './locales/ja.json';

const locales = { en, ja };

const I18nContext = createContext();

// ブラウザの言語設定を取得
const getBrowserLanguage = () => {
  const lang = navigator.language || navigator.userLanguage;
  // 'ja' または 'ja-JP' -> 'ja'
  // 'en' または 'en-US' -> 'en'
  const shortLang = lang.split('-')[0];
  return locales[shortLang] ? shortLang : 'en';
};

// ローカルストレージから言語設定を取得
const getStoredLanguage = () => {
  const stored = localStorage.getItem('language');
  return stored && locales[stored] ? stored : null;
};

export const I18nProvider = ({ children }) => {
  const [language, setLanguage] = useState(() => {
    return getStoredLanguage() || getBrowserLanguage();
  });

  useEffect(() => {
    localStorage.setItem('language', language);
    document.documentElement.lang = language;
  }, [language]);

  const t = (key, params = {}) => {
    const keys = key.split('.');
    let value = locales[language];
    
    for (const k of keys) {
      if (value && typeof value === 'object' && k in value) {
        value = value[k];
      } else {
        // フォールバック: 英語を試す
        value = locales.en;
        for (const fallbackKey of keys) {
          if (value && typeof value === 'object' && fallbackKey in value) {
            value = value[fallbackKey];
          } else {
            return key; // キーが見つからない場合はキーをそのまま返す
          }
        }
        break;
      }
    }

    if (typeof value !== 'string') {
      return key;
    }

    // パラメータの置換 {{param}}
    return value.replace(/\{\{(\w+)\}\}/g, (match, paramKey) => {
      return params[paramKey] !== undefined ? params[paramKey] : match;
    });
  };

  const changeLanguage = (lang) => {
    if (locales[lang]) {
      setLanguage(lang);
    }
  };

  const value = {
    language,
    changeLanguage,
    t,
    availableLanguages: Object.keys(locales),
  };

  return (
    <I18nContext.Provider value={value}>
      {children}
    </I18nContext.Provider>
  );
};

export const useI18n = () => {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
};

export default I18nContext;
