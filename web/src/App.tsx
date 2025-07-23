import { HomepageLayout } from '@/components/layout';
import { 
  HeroSection, 
  FeaturesSection, 
  HowItWorksSection, 
  StatsSection 
} from '@/components/sections';

function App() {
  return (
    <HomepageLayout>
      {/* hero section */}
      <HeroSection />
      
      {/* features section */}
      <FeaturesSection />
      
      {/* how it works section */}
      <HowItWorksSection />
      
      {/* stats and testimonials section */}
      <StatsSection />
    </HomepageLayout>
  );
}

export default App;