using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace AutoNeuralNet
{
    class Program
    {
        public static string ToString(double[] a) => "{" + string.Join(", ", a.Select(i => i.ToString())) + "}";
        static void Main(string[] args)
        {
            var net = new NeuralCore(@"C:\Users\Mikhail\source\repos\ANNGit\AutoNeuralNet\AutoNeuralNet\links.txt");
            //net.SetAllMatrixes(new double[][,] { new double[,] { { -0.2, 0.1 }, { 0.3, -0.3 }, { -0.4, -0.4 } }, new double[,] { { 0.2 }, { 0.3 } } });
            double[][] inputs = new double[4][];
            double[][] outs = new double[4][];
            StreamReader stream = new StreamReader(@"C:\Users\Mikhail\source\repos\ANNGit\AutoNeuralNet\AutoNeuralNet\Inputs.txt");
            for (int i = 0; i < 4; i++)
            {
                string[] pair = stream.ReadLine().Split(' ');
                inputs[i] = pair[0].ToArray().Select(j => double.Parse($"{j}")).ToArray();
                outs[i] = pair[1].ToArray().Select(j => double.Parse($"{j}")).ToArray();
            }
            //net.StartTraining(100000, inputs, outs, 0.01);
            //net.SaveLinks();
            for (int i = 0; i < inputs.Length; i++)
            {
                Console.WriteLine($"{ToString(  net.RunNet(inputs[i]))} => {ToString(outs[i])}");
            }
        }
    }
}
