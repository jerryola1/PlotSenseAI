import { useEffect } from 'react';
import { validateEnv } from '@/config/env';
import { cn } from '@/lib/utils';
import Header from './header';
import Footer from './footer';

interface HomepageLayoutProps {
  children: React.ReactNode;
  className?: string;
}

const HomepageLayout: React.FC<HomepageLayoutProps> = ({
  children,
  className,
}) => {
  useEffect(() => {
    try {
      validateEnv();
    } catch (error) {
      console.error('environment validation failed:', error);
    }
  }, []);

  return (
    <div className={cn('min-h-screen flex flex-col bg-background', className)}>
      <Header className="bg-transparent absolute top-0 left-0 right-0 z-50 border-none" />
      
      <main className="flex-1">
        {children}
      </main>
      
      <Footer />
    </div>
  );
};

export default HomepageLayout;