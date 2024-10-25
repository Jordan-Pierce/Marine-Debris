import os
import glob
import argparse
from tqdm import tqdm
import panoptes_client

import pandas as pd


# ----------------------------------------------------------------------------------------------------------------------
# Classes
# ----------------------------------------------------------------------------------------------------------------------

class ZooniverseUploader:
    def __init__(self, username, password, zoon_project_id, input_dir, set_active, upload, output_dir):
        """

        :param username:
        :param password:
        :param zoon_project_id:
        :param input_dir:
        :param set_active:
        :param upload:
        :param output_dir:
        """
        self.username = username
        self.password = password
        self.zoon_project_id = zoon_project_id
        self.input_dir = input_dir
        self.set_active = set_active
        self.upload = upload
        self.output_dir = output_dir.replace("\\", "/")
        self.client = None
        self.project = None
        self.dataframe = None

    def connect_to_zooniverse(self):
        """

        :return:
        """
        try:
            panoptes_client.Panoptes.connect(username=self.username, password=self.password)
            print(f"NOTE: Authentication to Zooniverse successful for {self.username}")
        except Exception as e:
            raise Exception(f"ERROR: Could not login to Panoptes for {self.username}. {str(e)}")

        try:
            self.project = panoptes_client.Project.find(id=self.zoon_project_id)
            print(f"NOTE: Connected to Zooniverse project '{self.project.title}' successfully")
        except Exception as e:
            raise Exception(f"ERROR: Could not access project {self.zoon_project_id}. {str(e)}")

    def prepare_dataframe(self):
        """

        :return:
        """
        tile_paths = glob.glob(f"{self.input_dir}/*.jpeg")[0:20]
        data = []

        for tile_path in tile_paths:
            mosaic_name = os.path.basename(tile_path).split("---")[0]
            tile_name = os.path.basename(tile_path)
            tile_id = tile_name.split("---")[-1].split(".")[0]

            data.append([mosaic_name, tile_name, tile_id, tile_path])

        self.dataframe = pd.DataFrame(data, columns=['Mosaic Name',
                                                     'Tile Name',
                                                     'Tile ID',
                                                     'Path'])

    def upload_to_zooniverse(self):
        """

        :return:
        """
        if not self.upload:
            return

        try:
            subject_set = panoptes_client.SubjectSet()
            subject_set.links.project = self.project
            subject_set.display_name = str(self.dataframe['Mosaic Name'].iloc[0])
            subject_set.save()

            self.project.reload()

            subject_dict = self.dataframe.to_dict(orient='records')
            subject_meta = {d['Path']: {k: v for k, v in d.items() if k != 'Path'} for d in subject_dict}

            subjects = []
            subject_ids = []

            for filename, metadata in tqdm(subject_meta.items()):
                subject = panoptes_client.Subject()
                subject.links.project = self.project
                subject.add_location(filename)
                subject.metadata.update(metadata)
                subject.save()
                subjects.append(subject)
                subject_ids.append(subject.id)

            subject_set.add(subjects)
            subject_set.save()
            self.project.save()

            if self.set_active:
                self.set_active_workflows(subject_set)

            self.dataframe['Subject_ID'] = subject_ids
            self.dataframe.to_csv(f"{self.input_dir}/frames.csv", index=False)

        except Exception as e:
            raise Exception(f"ERROR: Could not finish uploading subject set. {str(e)}")

    def set_active_workflows(self, subject_set):
        """

        :param subject_set:
        :return:
        """
        try:
            workflow_ids = self.project.__dict__['raw']['links']['active_workflows']

            for workflow_id in tqdm(workflow_ids):
                workflow = self.client.Workflow(workflow_id)
                workflow_name = workflow.__dict__['raw']['display_name']
                print(f"\nNOTE: Adding subject set {subject_set.display_name} to workflow {workflow_name}")
                workflow.add_subject_sets([subject_set])
                workflow.save()
                self.project.save()

        except Exception as e:
            raise Exception(f"ERROR: Could not link media to project workflows. {str(e)}")

    def run(self):
        """

        :return:
        """
        print("\n###############################################")
        print("Upload to Zooniverse")
        print("###############################################\n")

        os.makedirs(self.output_dir, exist_ok=True)

        self.connect_to_zooniverse()
        self.prepare_dataframe()
        self.upload_to_zooniverse()

        print("Done.")


# ----------------------------------------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Upload Data to Zooniverse")

    parser.add_argument("--username", type=str,
                        default=os.getenv('ZOONIVERSE_USERNAME'),
                        help="Zooniverse username")

    parser.add_argument("--password", type=str,
                        default=os.getenv('ZOONIVERSE_PASSWORD'),
                        help="Zooniverse password")

    parser.add_argument("--zoon_project_id", type=int, default=24250,
                        help="Zooniverse project ID")

    parser.add_argument("--input_dir", type=str,
                        help="Path to directory containing tiles")

    parser.add_argument("--set_active", action='store_true',
                        help="Make subject-set active with current workflow")

    parser.add_argument("--upload", action='store_true',
                        help="Upload media to Zooniverse (debugging)")

    parser.add_argument("--output_dir", type=str,
                        default=f'{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/Data',
                        help="Path to the output directory.")

    args = parser.parse_args()

    try:
        uploader = ZooniverseUploader(
            username=args.username,
            password=args.password,
            zoon_project_id=args.zoon_project_id,
            input_dir=args.input_dir,
            set_active=args.set_active,
            upload=args.upload,
            output_dir=args.output_dir
        )
        uploader.run()
    except Exception as e:
        print(f"ERROR: {e}")


if __name__ == "__main__":
    main()