#include "Solver/Solver.hpp"
#include "Model/ArealModel.hpp"
#include "Model/DiscreteModel.hpp"
#include "FileWriter/FileWriter.hpp"

#define NAN nc::constants::nan

int main()
{
    // auto* model = new ArealPreisachModel(1.0);
    // // auto* model = new DiscretePreisachModel(1.0, 0.01);
    // OscillatorParams params = {};
    // params.E = 1.35;
    // params.Preisach = model;
    //
    // auto [x, v] = OscillatorSolver(params);
    //
    // auto* xFile = new FileWriter("preisach_x.txt");
    // xFile->Write(x);
    //
    // auto* vFile = new FileWriter("preisach_v.txt");
    // vFile->Write(v);
    //
    // auto* uFile = new FileWriter("preisach_inputs.txt");
    // auto* lFile = new FileWriter("preisach_loop.txt");
    // auto [inputs, outputs] = model->HysteresisLoop();
    // uFile->Write(inputs);
    // lFile->Write(outputs);



    constexpr double time = 140.0;
    constexpr double dt = 0.1;
    constexpr auto steps = static_cast<uint32_t>(time / dt);
    std::vector<double> preisach_output(steps);
    std::vector<double> x(steps);

    for (uint32_t i = 0; i < steps; i++)
    {
        x[i] = 2 * nc::sin(i * dt);
    }

    std::vector<double> y(steps);
    y[0] = 0.5;
    // auto* model = new DiscretePreisachModel(1.0, 0.01);
    // // preisach_output[0] = 0.5;
    // for (uint32_t i = 1; i < steps; i++)
    // {
    //     const double p = model->P(x[i - 1]);
    //     preisach_output[i] = p;
    //     y[i] = y[i - 1] + p;
    // }
    
    preisach_output[0] = 0.5;
    bool isOverflowedUp = false;
    bool isOverflowedDown = false;
    for (uint32_t i = 1; i < steps; i++)
    {
        double p = 0.0;
        
        if (x[i] >= -1 && x[i] <= 1)
        {
            if (isOverflowedDown)
            {
                isOverflowedDown = false;
                p = std::pow(1 + x[i], 2) / 2;
            }
            else if (isOverflowedUp)
            {
                isOverflowedUp = false;
                p = std::pow(1 + x[i], 2) / 2 - 2;
            }
            else if (x[i] > x[i - 1])
            {
                p = (x[i] - x[i - 1]) * (x[i] + x[i - 1] + 2) / 2;
            }
            else if (x[i] < x[i - 1])
            {
                p = -( (x[i - 1] - x[i]) * (2 - x[i - 1] - x[i]) / 2 );
            }
        }
        else
        {
            if (x[i] < -1)
            {
                isOverflowedDown = true;
            }
            else if (x[i] > 1)
            {
                isOverflowedUp = true;
            }
            p = preisach_output[i - 1];
        }
    
        preisach_output[i] = p;
        y[i] = y[i - 1] + p;
    }
    
    auto* uFile = new FileWriter("preisach_inputs.txt");
    auto* lFile = new FileWriter("preisach_loop.txt");
    uFile->Write(x);
    lFile->Write(preisach_output);
    // lFile->Write(y);






    

    // auto* model = new ArealPreisachModel(1.0);
    // constexpr double time = 100.0;
    // constexpr double dt = 0.1;
    // constexpr auto steps = static_cast<uint32_t>(time / dt);
    // std::vector<double> x(steps);
    //
    // x[0] = 0.0;
    //
    // for (uint32_t i = 1; i < steps; i++)
    // {
    //     const double t = i * dt;
    //     const double force = nc::sin(t / 10.0) * nc::sin(t);
    //     x[i] = force + model->P(x[i - 1]);
    // }
    //
    // auto* sinFile = new FileWriter("preisach_sin.txt");
    // sinFile->Write(x);
    //
    // auto* uFile = new FileWriter("preisach_inputs.txt");
    // auto* lFile = new FileWriter("preisach_loop.txt");
    // auto [inputs, outputs] = model->HysteresisLoop();
    // uFile->Write(inputs);
    // lFile->Write(outputs);

    // std::vector vec1 = {0.7, 0.75, 0.8, NAN, NAN, NAN, 0.6, 0.725, NAN, NAN, -0.15, -0.05, 0.1, NAN};
    // std::vector vec2 = {NAN, NAN, NAN, -0.3, -0.25, -0.2, NAN, NAN, -0.1, -0.15, NAN, NAN, NAN, -0.05};
    // double x = -0.1872;
    //
    // auto bounds = FindBoundsIndices(vec2, x);
    //
    // for (const auto& [l, r] : bounds)
    // {
    //     std::cout << l << " " << r << '\n';
    // }
    
    return 0;
}
