import { Card, CardContent } from '@/components/ui/card';
import { Button } from '@/components/ui';
import Container from '@/components/layout/container';
import { EXTERNAL_LINKS } from '@/constants';

interface Step {
  number: string;
  title: string;
  description: string;
  icon: React.ReactNode;
  color: string;
}

const steps: Step[] = [
  {
    number: '01',
    title: 'load your dataframe',
    description: 'import plotsense and load your pandas dataframe. works with any dataset loaded via pandas from csv, json, xlsx, or database connections.',
    color: 'from-blue-500 to-blue-600',
    icon: (
      <svg className="h-10 w-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
      </svg>
    ),
  },
  {
    number: '02',
    title: 'get ai recommendations',
    description: 'call ps.recommender(df) to analyze your dataframe using ensemble ai models. returns ranked suggestions with ensemble scores and model agreement metrics.',
    color: 'from-purple-500 to-purple-600',
    icon: (
      <svg className="h-10 w-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
      </svg>
    ),
  },
  {
    number: '03',
    title: 'generate matplotlib plots',
    description: 'use ps.plotgen(df, suggestions) to create publication-ready matplotlib figures. supports customization and returns figure objects for further modification.',
    color: 'from-green-500 to-green-600',
    icon: (
      <svg className="h-10 w-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
      </svg>
    ),
  },
  {
    number: '04',
    title: 'get ai explanations',
    description: 'call ps.explainer(fig) to receive detailed markdown explanations. ai analyzes your plot and provides insights, patterns, and statistical interpretations.',
    color: 'from-orange-500 to-orange-600',
    icon: (
      <svg className="h-10 w-10" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
      </svg>
    ),
  },
];

const HowItWorksSection: React.FC = () => {
  return (
    <section className="py-20 bg-white dark:bg-gray-800">
      <Container>
        <div className="text-center mb-16 animate-fade-in">
          <div className="inline-flex items-center px-4 py-2 rounded-full bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-sm font-medium mb-4">
            how it works
          </div>
          <h2 className="text-4xl md:text-5xl font-bold text-gray-900 dark:text-gray-100 mb-6">
            from dataframe to insights in
            <span className="block text-blue-600 dark:text-blue-400">4 simple steps</span>
          </h2>
          <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
            plotsense integrates seamlessly with your existing python data science workflow, 
            making ai-powered visualization accessible to all skill levels.
          </p>
        </div>

        {/* steps grid */}
        <div className="grid lg:grid-cols-4 gap-8 mb-16">
          {steps.map((step, index) => (
            <div key={index} className="relative animate-fade-in" style={{ animationDelay: `${index * 0.2}s` }}>
              {/* connector line */}
              {index < steps.length - 1 && (
                <div className="hidden lg:block absolute top-1/2 left-full w-full h-0.5 bg-gradient-to-r from-gray-200 to-transparent dark:from-gray-600 transform -translate-y-1/2 z-0">
                  <div className="absolute right-0 top-1/2 w-2 h-2 bg-gray-300 dark:bg-gray-600 rounded-full transform -translate-y-1/2"></div>
                </div>
              )}
              
              <Card className="relative z-10 group hover:shadow-lg transition-all duration-300 border-2 border-gray-200 dark:border-gray-700 hover:border-blue-500/20 dark:hover:border-blue-400/20 bg-white dark:bg-gray-800">
                <CardContent className="p-8 text-center">
                  <div className="space-y-6">
                    {/* step number */}
                    <div className="relative">
                      <div className={`inline-flex items-center justify-center w-16 h-16 rounded-full bg-gradient-to-r ${step.color} text-white font-bold text-xl mb-4 group-hover:scale-110 transition-transform duration-300`}>
                        {step.number}
                      </div>
                    </div>
                    
                    {/* icon */}
                    <div className={`inline-flex items-center justify-center w-12 h-12 rounded-lg bg-gradient-to-r ${step.color} text-white`}>
                      {step.icon}
                    </div>
                    
                    {/* content */}
                    <div className="space-y-3">
                      <h3 className="text-xl font-semibold text-gray-900 dark:text-gray-100 capitalize">
                        {step.title}
                      </h3>
                      <p className="text-gray-600 dark:text-gray-300 leading-relaxed text-sm">
                        {step.description}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          ))}
        </div>

        {/* demo section */}
        <div className="bg-gradient-to-r from-gray-50 to-blue-50 dark:from-gray-800 dark:to-blue-900/20 rounded-3xl p-8 lg:p-12 text-center animate-fade-in">
          <div className="max-w-2xl mx-auto space-y-6">
            <div className="inline-flex items-center px-4 py-2 rounded-full bg-blue-100 dark:bg-blue-900/50 text-blue-700 dark:text-blue-300 text-sm font-medium">
              see it in action
            </div>
            
            <h3 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
              ready to enhance your data analysis?
            </h3>
            
            <p className="text-lg text-gray-600 dark:text-gray-300">
              join data scientists and researchers who use plotsense to accelerate 
              their exploratory data analysis with ai-powered insights.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Button 
                size="lg" 
                className="px-8 py-4 text-lg font-semibold"
                onClick={() => window.open(EXTERNAL_LINKS.BINDER_DEMO + '?utm_source=website&utm_medium=how_it_works&utm_campaign=try_now', '_blank')}
              >
                try it now
                <svg className="ml-2 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-2M7 17l3-3m0 0l3-3m-3 3h12" />
                </svg>
              </Button>
              
              <Button 
                variant="outline" 
                size="lg" 
                className="px-8 py-4 text-lg font-semibold"
                onClick={() => window.open(EXTERNAL_LINKS.GITHUB_REPO + '?utm_source=website&utm_medium=how_it_works&utm_campaign=view_examples', '_blank')}
              >
                view examples
                <svg className="ml-2 h-5 w-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
                </svg>
              </Button>
            </div>
          </div>
        </div>
      </Container>
    </section>
  );
};

export default HowItWorksSection;