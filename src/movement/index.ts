/**
 * Local-movement learning subsystem (standing objective #2).
 *
 * A schema for low-level mouse/keyboard/window events, a seeded synthetic
 * stream generator, a JSONL dataset codec, a pluggable model backend that can
 * repeat *and* generalize recorded movements, and a generalization eval harness.
 * All deterministic and dependency-free so the full pipeline runs in CI without
 * a real OS; the model backend is the seam for a real on-device small model.
 */

export {
  buildMovementDataset,
  buildMovementDemonstration,
  datasetIntents,
  euclideanDistance,
  isPositionedEvent,
  POINTER_BUTTONS,
  pointOf,
  pointerPath,
  sortEvents,
  type MovementDataset,
  type MovementDemonstration,
  type MovementEvent,
  type MovementEventKind,
  type Point,
  type PointerButton,
} from "./events.js";
export {
  retargetEvents,
  SimilarityTransformBackend,
  similarityTransform,
  type MovementContext,
  type MovementModelBackend,
  type TrainedMovementModel,
} from "./model.js";
export {
  createSeededRandom,
  generateDragDemonstration,
  generateSyntheticDataset,
  type DragShape,
  type SyntheticDatasetParams,
  type SyntheticDragParams,
} from "./synthetic.js";
export { decodeDatasetJsonl, encodeDatasetJsonl } from "./dataset.js";
export {
  evaluateCase,
  evaluateGeneralization,
  pathRmse,
  resamplePath,
  type MovementEvalCase,
  type MovementEvalResult,
  type MovementEvalSummary,
} from "./eval.js";
