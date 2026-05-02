import { UploadCloud } from "lucide-react";
import { useCallback, useRef, useState } from "react";

interface CycleUploaderProps {
  onUpload: (file: File) => Promise<void>;
}

export function CycleUploader({ onUpload }: CycleUploaderProps): JSX.Element {
  const inputRef = useRef<HTMLInputElement | null>(null);
  const [dragging, setDragging] = useState(false);

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file) {
        return;
      }
      void onUpload(file);
    },
    [onUpload],
  );

  return (
    <section
      className={`rounded-lg border border-dashed p-5 transition ${
        dragging
          ? "border-emerald-400 bg-emerald-400/10"
          : "border-slate-700 bg-slate-900/70"
      }`}
      onDragOver={(event) => {
        event.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault();
        setDragging(false);
        handleFile(event.dataTransfer.files[0]);
      }}
    >
      <button
        type="button"
        className="flex w-full flex-col items-center gap-3 text-center text-slate-200"
        onClick={() => inputRef.current?.click()}
        aria-label="Upload cycle file"
      >
        <UploadCloud className="h-8 w-8 text-emerald-400" aria-hidden="true" />
        <span className="text-sm font-semibold">Drop cycle CSV or Parquet</span>
        <span className="text-xs leading-5 text-slate-400">
          Files are parsed by the backend; each curve must have 100 points.
        </span>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept=".csv,.parquet"
        className="hidden"
        onChange={(event) => handleFile(event.target.files?.[0])}
        aria-label="Browse cycle file"
      />
    </section>
  );
}
