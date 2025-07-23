import { cn } from '@/lib/utils';
import type { BaseComponent } from '@/types';

interface ContainerProps extends BaseComponent {
  size?: 'sm' | 'md' | 'lg' | 'xl' | 'full';
  center?: boolean;
}

const Container: React.FC<ContainerProps> = ({
  children,
  className,
  size = 'lg',
  center = true,
}) => {
  const sizeClasses = {
    sm: 'max-w-2xl',
    md: 'max-w-4xl',
    lg: 'max-w-5xl',
    xl: 'max-w-6xl',
    full: 'max-w-full',
  };

  return (
    <div
      className={cn(
        'w-full px-4 sm:px-6',
        sizeClasses[size],
        center && 'mx-auto',
        className
      )}
    >
      {children}
    </div>
  );
};

export default Container;