// LLM-rApp + deterministic xApp scenario, 20 vehicles.
#include "../../../oran_xapp/_common/scenario_lib.h"

int main(int argc, char* argv[]) {
    oran_xapp::ScenarioConfig cfg;
    cfg.method = oran_xapp::METHOD_M4_XAPP_PROPOSED;
    cfg.scenarioName = "llm_v2x_20cars";
    cfg.runLabel = "llm_v2x_s20";
    cfg.ueNum = 20;
    cfg.numGnbs = 1;
    cfg.ns2TraceFile = "scratch/mobility_1/ns2manhattanmobility20.tcl";
    cfg.gnbPositions = { ns3::Vector(300, 300, 25) };
    cfg.simTimeSeconds = 5.0;
    cfg.useShm = true;
    cfg.allowAdaptiveStudy = true;
    cfg.assignUseCase = oran_xapp::MakeAssignByMix(20, 1, 5, 1);
    cfg.heuristicControl = &oran_xapp::HeuristicM4;
    return oran_xapp::RunScenario(cfg, argc, argv);
}

