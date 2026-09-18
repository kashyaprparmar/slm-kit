import { TrainingStudio } from "@/components/studio/TrainingStudio";

export default function Alignment() {
  return (
    <TrainingStudio
      task="alignment"
      title="Preference Alignment"
      description="Run DPO, IPO, ORPO, SimPO, or KTO through the shared preference-training pipeline."
      datasetKinds={["preference", "kto"]}
      defaultOutputName="my-aligned-model"
      note="PPO is intentionally unavailable. Reference-free objectives do not allocate a reference model."
    />
  );
}
