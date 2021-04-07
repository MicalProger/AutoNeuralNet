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
        public static string ToString(double[] a) => "{" + string.Join(", ", a.Select(i => Math.Round(i, 2).ToString())) + "}";
        static void Main(string[] args)
        {
            var net = new NeuralCore(new int[] {225, , 10});
            //net.SetAllMatrixes(new double[][,] { new double[,] { { -0.2, 0.1 }, { 0.3, -0.3 }, { -0.4, -0.4 } }, new double[,] { { 0.2 }, { 0.3 } } });
            double[][] inputs = new double[19][];
            double[][] outs = new double[19][];
            StreamReader stream = new StreamReader(@"C:\Users\RazRab\source\repos\ANNGit\AutoNeuralNet\AutoNeuralNet\Inputs.txt");
            for (int i = 0; i < 19; i++)
            {
                string[] pair = stream.ReadLine().Split(' ');
                inputs[i] = pair[0].Split('|').Select(j => double.Parse($"{ j}")).ToArray();
                outs[i] = pair[1].ToArray().Select(j => double.Parse($"{j}")).ToArray();
            }
            net.StartTraining(100000, inputs, outs, 0.001);
            //net.SaveLinks();
            for (int i = 0; i < inputs.Length; i++)
            {
                Console.WriteLine($"{ToString(  net.RunNet(inputs[i]))} => {ToString(outs[i])}");
                
            }
            //Console.WriteLine($"{ToString(net.RunNet(new double[] {1, 0, 0, 0 }))}");
        }
    }
}
