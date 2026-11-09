// The site's side of scripts/engine_parity.py: runs jordannedyck-ai-web's own
// lib/*.ts on the fixture file and writes what each function returns.
//
//   node engine_parity.mjs <web lib dir> <fixtures.json> <out.json>
//
// Node 22.18+ loads the .ts files with its built-in type stripping. classify.ts
// imports the AI SDK, zod and the database pool at load; they are stubbed
// here, since freeLabels() uses none of them.
import { register } from 'node:module';
import { readFileSync, writeFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

const [libDir, fixturesPath, outPath] = process.argv.slice(2);

const STUBS = {
  ai: 'export const generateText = () => {}; export const Output = {};',
  '@ai-sdk/openai': 'export const openai = () => {};',
  zod: 'const p = new Proxy(function () {}, { get: () => p, apply: () => p }); export const z = p;',
  pg: 'export class Pool {}',
};
const loader = `
const STUBS = ${JSON.stringify(STUBS)};
export async function resolve(specifier, context, next) {
  if (STUBS[specifier] !== undefined) {
    return { url: 'data:text/javascript,' + encodeURIComponent(STUBS[specifier]), shortCircuit: true };
  }
  if (specifier.startsWith('.') && !/\\.[cm]?[jt]s$/.test(specifier)) {
    return next(specifier + '.ts', context);
  }
  return next(specifier, context);
}`;
register('data:text/javascript,' + encodeURIComponent(loader));

const lib = (name) => import(pathToFileURL(path.join(libDir, name + '.ts')).href);
const citations = await lib('citations');
const projects = await lib('projects');
const timeline = await lib('timeline');
const cost = await lib('cost');
const promptRequest = await lib('prompt-request');
const classify = await lib('classify');
const buildStory = await lib('build-story');

// stripCitationsTransform() over the given deltas: the text-delta texts it emits.
async function streamStrip(deltas) {
  const sink = { raw: '' };
  const ts = citations.stripCitationsTransform(sink)();
  const writer = ts.writable.getWriter();
  const out = [];
  const reading = (async () => {
    for await (const part of ts.readable) if (part.type === 'text-delta') out.push(part.text);
  })();
  for (const text of deltas) await writer.write({ type: 'text-delta', id: 't', text });
  await writer.close();
  await reading;
  return { out, raw: sink.raw };
}

const fx = JSON.parse(readFileSync(fixturesPath, 'utf8'));
const result = { answers: [], questions: [], sections: [], costs: [] };

for (const a of fx.answers) {
  const { segments, citedIds } = citations.parseCitations(a.answer_cited);
  const visible = timeline.withoutTimelineMarker(citations.stripCitations(a.answer_cited));
  result.answers.push({
    segments,
    citedIds,
    sources: citations.buildSources(segments, a.chunks),
    labels: citations.citationLabels(segments, a.chunks),
    project: projects.pickProject(citedIds, a.chunks),
    predicted: projects.predictProject(a.chunks),
    citesCareer: timeline.citesCareer(citedIds),
    leadsWithCareer: timeline.leadsWithCareer(a.chunks),
    timelineFor: timeline.timelineFor(a.query, citedIds, a.chunks),
    freeLabels: classify.freeLabels(a.query, visible),
    stream: await streamStrip(a.deltas),
  });
}
for (const q of fx.questions) {
  result.questions.push({
    instructions: promptRequest.asksForInstructions(q),
    overview: timeline.asksForOverview(q),
    timeline: timeline.asksForTimeline(q),
    careerHistory: timeline.asksForCareerHistory(q),
    replay: timeline.isTimelineReplay(q, true),
    timelineForUncited: timeline.timelineFor(q, [], []),
    buildStory: buildStory.asksForBuildStory(q),
    storyFromCard: buildStory.storyFromCard('ai-resume', q),
    freeLabelsQueryOnly: classify.freeLabels(q, null),
  });
}
for (const s of fx.sections) result.sections.push(citations.sectionOf(s.content, s.title));
for (const c of fx.costs) result.costs.push(cost.estimateCostUsd(c.model, c.prompt, c.completion));

writeFileSync(outPath, JSON.stringify(result));
