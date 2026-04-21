export default function App() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-slate-950 text-slate-100">
      <div
        className="absolute inset-0 opacity-20"
        style={{
          backgroundImage:
            "linear-gradient(to right, rgba(255,255,255,0.18) 1px, transparent 1px), linear-gradient(to bottom, rgba(255,255,255,0.18) 1px, transparent 1px)",
          backgroundSize: "min(8vw, 56px) min(8vw, 56px)",
        }}
      />

      <main className="relative mx-auto flex min-h-screen w-full max-w-5xl items-center px-6 py-16">
        <div className="w-full space-y-10">
          <div className="space-y-4">
            <p className="text-sm font-medium uppercase tracking-[0.2em] text-indigo-300">Maia Chess</p>
            <h1 className="text-4xl font-semibold tracking-tight text-balance md:text-6xl">
              REST API for human-like move prediction from FEN.
            </h1>
            <p className="max-w-2xl text-base leading-relaxed text-slate-300 md:text-lg">
              FastAPI service focused on policy-network inference only. Input a FEN, return top legal UCI moves
              with probabilities, and optional temperature scaling for human-like behavior.
            </p>
          </div>

          <div className="grid gap-8 md:grid-cols-2">
            <section className="space-y-3">
              <h2 className="text-lg font-semibold text-indigo-200">POST /analyze</h2>
              <pre className="overflow-x-auto border border-white/20 bg-black/30 p-4 text-sm leading-relaxed text-slate-200">
{`{
  "fen": "r1bqkbnr/pppp1ppp/2n5/4p3/3P4/5N2/PPP1PPPP/RNBQKB1R b KQkq - 2 3",
  "top_k": 5,
  "temperature": 1.0
}`}
              </pre>
            </section>

            <section className="space-y-3">
              <h2 className="text-lg font-semibold text-indigo-200">Response</h2>
              <pre className="overflow-x-auto border border-white/20 bg-black/30 p-4 text-sm leading-relaxed text-slate-200">
{`{
  "moves": [
    { "uci": "g8f6", "prob": 0.32 },
    { "uci": "d7d5", "prob": 0.21 }
  ]
}`}
              </pre>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}