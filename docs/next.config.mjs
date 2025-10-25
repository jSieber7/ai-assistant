import nextra from 'nextra'

const withNextra = nextra({
  theme: 'nextra-theme-docs',
  themeConfig: './theme.config.js',
  flexsearch: {
    codeblocks: false
  }
})

export default withNextra({
  // Your Next.js config here
  output: 'export',
  trailingSlash: true,
  images: {
    unoptimized: true
  },
  // Configure base path for GitHub Pages
  basePath: process.env.NODE_ENV === 'production' ? '/ai_assistant' : '',
  assetPrefix: process.env.NODE_ENV === 'production' ? '/ai_assistant' : ''
})