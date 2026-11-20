// LLM-rApp + deterministic xApp scenario, 30 vehicles.
#include "../../../oran_xapp/_common/scenario_lib.h"

int main(int argc, char* argv[]) {
    oran_xapp::ScenarioConfig cfg;
    cfg.method = oran_xapp::METHOD_M4_XAPP_PROPOSED;
    cfg.scenarioName = "llm_v2x_30cars";
    cfg.runLabel = "llm_v2x_s30";
    cfg.ueNum = 30;
    cfg.numGnbs = 1;
    cfg.ns2TraceFile = "scratch/mobility_1/ns2manhattanmobility30.tcl";
    cfg.gnbPositions = { ns3::Vector(300, 300, 25) };
    cfg.simTimeSeconds = 5.0;
    cfg.useShm = true;
    cfg.allowAdaptiveStudy = true;
    cfg.assignUseCase = oran_xapp::MakeAssignByMix(30, 3, 7, 3);
    cfg.heuristicControl = &oran_xapp::HeuristicM4;
    return oran_xapp::RunScenario(cfg, argc, argv);
}

