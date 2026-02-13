import React from 'react';
import { useI18n } from '../i18n';
import './LanguageSwitcher.css';

const LanguageSwitcher = () => {
  const { language, changeLanguage, t } = useI18n();

  const handleLanguageChange = (lang) => {
    changeLanguage(lang);
  };

  return (
    <div className="language-switcher">
      <button
        className={`lang-btn ${language === 'en' ? 'active' : ''}`}
        onClick={() => handleLanguageChange('en')}
        title={t('language.en')}
      >
        EN
      </button>
      <span className="lang-separator">|</span>
      <button
        className={`lang-btn ${language === 'ja' ? 'active' : ''}`}
        onClick={() => handleLanguageChange('ja')}
        title={t('language.ja')}
      >
        JA
      </button>
    </div>
  );
};

export default LanguageSwitcher;
