export default {
  project: {
    link: 'https://github.com/jSieber7/ai_assistant'
  },
  docsRepositoryBase: 'https://github.com/jSieber7/ai_assistant/tree/main/docs/pages',
  titleSuffix: ' – AI Assistant Documentation',
  getNextSeoProps: () => ({
    titleTemplate: '%s – AI Assistant Documentation'
  }),
  navigation: true,
  darkMode: true,
  search: {
    codeblocks: false
  },
  footer: {
    text: `MIT ${new Date().getFullYear()} © AI Assistant.`
  },
  editLink: {
    text: 'Edit this page on GitHub'
  },
  logo: (
    <>
      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
      </svg>
      <span style={{ marginLeft: '.4em', fontWeight: 'bold' }}>AI Assistant</span>
    </>
  ),
  head: (
    <>
      <meta name="viewport" content="width=device-width, initial-scale=1.0" />
      <meta name="description" content="Comprehensive documentation for AI Assistant - a powerful multi-agent system with LangChain integration" />
      <meta name="og:title" content="AI Assistant Documentation" />
      <meta name="og:description" content="Comprehensive documentation for AI Assistant - a powerful multi-agent system with LangChain integration" />
    </>
  ),
  sidebar: {
    defaultMenuCollapseLevel: 2
  },
  toc: {
    backToTop: true
  }
}