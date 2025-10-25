import { Footer, Layout, Navbar } from 'nextra-theme-docs'
import { Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'

export const metadata = {
  // Define your metadata here
  // For more information on metadata API, see: https://nextjs.org/docs/app/building-your-application/optimizing/metadata
}

const banner = <div style={{ padding: '12px 24px', background: '#f0f9ff', border: '1px solid #bae6fd', borderRadius: '8px', margin: '16px 0' }}>
  <strong>🎉 Welcome to AI Assistant Documentation!</strong> This is the new Nextra-based documentation site. Explore the comprehensive guides and API references.
</div>

const navbar = (
  <Navbar
    logo={<b>AI Assistant</b>}
    // ... Your additional navbar options
  />
)

const footer = <Footer>MIT {new Date().getFullYear()} © AI Assistant.</Footer>

export default async function RootLayout({ children }) {
  return (
    <html
      // Not required, but good for SEO
      lang="en"
      // Required to be set
      dir="ltr"
      // Suggested by `next-themes` package https://github.com/pacocoursey/next-themes#with-app
      suppressHydrationWarning
    >
      <Head
      // ... Your additional head options
      >
        {/* Your additional tags should be passed as `children` of `<Head>` element */}
      </Head>
      <body>
        <Layout
          banner={banner}
          navbar={navbar}
          pageMap={await getPageMap()}
          docsRepositoryBase="https://github.com/your-username/ai-assistant/tree/main/docs"
          footer={footer}
          // ... Your additional layout options
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}