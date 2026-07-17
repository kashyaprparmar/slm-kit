import { TrainingStudio } from "@/components/studio/TrainingStudio";

export default function DomainAdaptation() {
  return (
    <TrainingStudio
      task="continued_pretrain"
      title="Domain Adaptation Studio"
      description="Continue-pretrain a base model on a domain corpus, then reuse it as a base."
      datasetKinds={["domain_corpus"]}
      defaultOutputName="my-domain-base"
      note="On completion this model is saved as a reusable base — pick it (via local path or after publishing to HF) in the Fine-Tuning Studio."
    />
  );
}
