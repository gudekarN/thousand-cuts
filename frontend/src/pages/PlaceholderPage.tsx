import { FlaskConical } from 'lucide-react'

export default function PlaceholderPage({ title }: { title: string }) {
  return (
    <div className="flex-1 flex items-center justify-center min-h-[60vh]">
      <div className="text-center space-y-3 max-w-sm">
        <div className="flex justify-center">
          <div className="p-3 bg-primary/10 rounded-full text-primary">
            <FlaskConical className="w-7 h-7" />
          </div>
        </div>
        <h2 className="text-lg font-semibold">{title}</h2>
        <p className="text-sm text-muted-foreground">
          This page is coming in a future task. The backend API is connected and ready.
        </p>
      </div>
    </div>
  )
}
