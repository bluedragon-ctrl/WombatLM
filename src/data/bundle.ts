/**
 * The data files shipped with the app. Vite bundles these JSON files; Node
 * scripts and tests import them the same way.
 */
import vocabSpec from '../../data/vocab.json';
import languages from '../../data/languages.json';
import forms from '../../data/forms.json';
import base from '../../data/animals.base.json';
import ocean from '../../data/packs/ocean.json';
import insects from '../../data/packs/insects.json';
import farmBabies from '../../data/packs/farm-babies.json';
import type { DataBundle, Pack } from './dataset';
import type { LanguageSpec, VocabSpec } from '../nn/vocab';

export const BASE_PACK = base as Pack;

/** Extension packs for Teach mode, by pack id. */
export const EXTRA_PACKS: Record<string, Pack> = Object.fromEntries(
  ([ocean, insects, farmBabies] as Pack[]).map((p) => [p.pack, p]),
);

export function baseBundle(): DataBundle {
  return {
    vocabSpec: vocabSpec as VocabSpec,
    languages: languages as LanguageSpec[],
    forms: forms as string[],
    packs: [BASE_PACK],
  };
}

/** Every pack shipped in the repository (for validation). */
export function allPacks(): Pack[] {
  return [BASE_PACK, ...Object.values(EXTRA_PACKS)];
}
